"""
    database/loader.py
    ------------------
    Loader object, retrieves specific data structures for UAV trajectory 
    defined in a particular manner.
"""
import numpy as np
import pandas as pd
import multiprocessing as mp

from pathlib import Path
from typing import Dict, Final, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from logs.logger import Logger, LogLevel
from database.file_handler import HandlerFactory, BaseFileHandler
from database.data_processor import DataProcessor


# ---------------========== Data Loader ==========--------------- #

class DataLoader:
    """
    High-level data loader that abstract file management and 
    data processing to the `database.file_hander.py` and 
    `database.data_processing.py` respectively. Remaining for
    this class, are the responsibility:
        - Loading datasets in chunks through file-handlers (e.g., csv)
        - Processing the data chunks into structured NumPy arrays
        - Saving processed data to memory.
        - Performing utility operations.
    """
    REQUIRED_COLUMNS: Final[List[str]] = [
        'dvec', 'rx_type', 'link_state', 'los_pl',
        'los_ang', 'los_dly', 'nlos_pl', 'nlos_ang', 'nlos_dly'
    ]

    def __init__(self,
        n_workers: Optional[int]=None, chunk_size: int=10_000,
        level: LogLevel=LogLevel.INFO, use_console: bool=True
    ):
        """ Initialize Data-Loader Instance """
        # Creates root directory for the data-loader
        self.directory = Path(__file__).parent / "storage"
        self.directory.mkdir(parents=True, exist_ok=True)

        self.n_workers = n_workers or mp.cpu_count()
        self.chunk_size = int(chunk_size)

        self.logger = Logger(
            "data-loader", level=level, json_format=True, use_console=use_console
        )
        self.processor = DataProcessor(self.logger)
    

    # -------------------- Save / Load -------------------- #

    def save(self,
        data: Dict[str, np.ndarray], filepath: Union[str, Path], fmt: str="csv"
    ) -> None:
        """
        Save structured data using the appropriate file handler.

        Args:
            data:   Processed dataset to store.
            filepath:   File name (without extension).
            fmt:    Output format (e.g., 'csv').
        """
        filepath = Path(self.directory) / filepath
        filepath = filepath.with_suffix(f".{fmt.lower()}")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        handler = HandlerFactory.get_handler_by_format(fmt, self.logger)
        handler.save(data, filepath)

    def load(self, filepaths: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Load and process one or more datasets into structured NumPy arrays.

        Args:
            filepaths (Union[str, List[str]]): File or list of file paths.

        Returns:
            Dict[str, np.ndarray]: Combined dataset keyed by column name.
        """
        filepaths = [filepaths] if isinstance(filepaths, (str,Path)) else filepaths
        self.logger.info(
            f"Loading {len(filepaths)} file(s) using {self.n_workers} worker(s)..."
        )
        
        all_chunks: List[pd.DataFrame] = []
        for filepath in filepaths:
            path = Path(self.directory) / filepath
            if not path.exists():
                self.logger.error(f"File not found: {path}")
                continue

            handler = HandlerFactory.get_handler(path, self.logger)
            chunks = handler.load_chunks(path, self.chunk_size)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise RuntimeError("No data chunks were successfully loaded.")

        # Choose executor type
        use_threads = any(Path(fp).suffix.lower() in ['.csv'] for fp in filepaths)
        Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

        # Parallel chunk processing
        with Executor(max_workers=self.n_workers) as executor:
            results = list(executor.map(self.processor.process_chunk, all_chunks))

        processed = self.processor.concatenate_results(results)
        self.logger.info(
            f"Loaded and processed {len(processed)} columns "
            f"from {len(filepaths)} file(s)."
        )
        return processed



# ---------------========== Utility Functions ==========--------------- #

def shuffle_and_split(
    data: Dict[str, np.ndarray], val_ratio: float = 0.20, seed: int = 42
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Randomly shuffle and split dataset into training and validation sets.

    Args:
        data:   Full dataset.
        val_ratio:  Fraction of validation data.
        seed:   RNG seed for reproducibility.

    Returns:
        (train_data, val_data)
    """
    lengths = {len(value) for value in data.values()}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent array lengths detected: {lengths}")
    n = lengths.pop()

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    split_idx = int(n * (1 - val_ratio))

    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    return (
        {key: value[train_idx] for key, value in data.items()},
        {key: value[val_idx] for key, value in data.items()},
    )
