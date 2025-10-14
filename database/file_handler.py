"""
    database/file_handler.py
    ------------------------
    Class for managing file processing, modular approach to enable additional
    file-format support without altering existing code.

    Supports:
        - CSV
"""
import orjson
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, List, Protocol, Union
from abc import ABC, abstractmethod

import pyarrow as pa
import pyarrow.csv as pv

from logs.logger import Logger


# ---------------========== Base Interfaces ==========--------------- #

class FileHandler(Protocol):
    """
    Interface definitions for file format handler
    """
    def load_chunks(self, filepath: Path, chunk_size: int) -> List[pd.DataFrame]:

        """
        Load file in `DataFrame` chunks
        """
        ...
    
    def save(self, data: Dict[str, np.ndarray], filepath: Path) -> None:
        """Save structured data to a file."""
        ...


class BaseFileHandler(ABC):
    """
    Abstract base class defining shared API and logger support for 
    all classes inherent this api. 
    """
    def __init__(self, logger: Logger):
        self.logger = logger
    

    @abstractmethod
    def load_chunks(self, filepath: Path, chunk_size: int) -> List[pd.DataFrame]:
        """Load file into DataFrame chunks."""
        pass


    @abstractmethod
    def save(self, data: Dict[str, np.ndarray], filepath: Path) -> None:
        """Save structured data to a file."""
        pass


    def _prepare_dataframe(self, data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Helper: Convert dict of numpy arrays to DataFrame.
        Converts object/nested arrays to list for serialization.
        """
        df_dict = {}
        for key, arr in data.items():
            if arr.dtype == object:
                df_dict[key] = [
                    v.tolist() if isinstance(v, np.ndarray) else v for v in arr
                ]
            else:
                df_dict[key] = arr
        return pd.DataFrame(df_dict)


# ---------------========== CSV Handler Implementation ==========--------------- #

class CsvHandler(BaseFileHandler):
    """
    Csv Handler implemented using PyArrow for efficient chunked I/O
    operations and being reliable for larger datasets.
    """

    def load_chunks(self, filepath: Path, chunk_size: int) -> List[pd.DataFrame]:
        """
        Load CSV file in chunks using `PyArrow RecordBatchReader`.
        Converts each batch to pandas for processing.

        Args:
            filepath:   Path to the CSV file.
            chunk_size: Approximate number of rows per chunk.

        Returns:
            Loaded DataFrame chunks.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file `{filepath}` not found")

        chunks: List[pd.DataFrame] = []
        try:
            # block_size is bytes, not rows, convert approximately
            block_size = max(1 << 20, chunk_size * 1024)  # at least 1MB
            reader = pv.open_csv(
                filepath,
                read_options=pv.ReadOptions(block_size=block_size, use_threads=True)
            )

            for batch in reader: chunks.append(batch.to_pandas())
            reader.close()
            self.logger.debug(f"Loaded {len(chunks)} chunks from CSV: {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to load CSV `{filepath}`: {e}")
            raise

        return chunks

    def save(self, data: Dict[str, np.ndarray], filepath: Path) -> None:
        """
        Save structured data to CSV file.

        Args:
            data:   Data to save.
            filepath:   Output CSV path.
        """
        df = self._prepare_dataframe(data)

        try:
            # Using Arrow write_csv for better performance than pandas
            table = pa.Table.from_pandas(df)
            pv.write_csv(table, filepath)
            self.logger.info(f"Saved `{len(df)}` rows to CSV: {filepath}")
        except Exception as e:
            # Fallback to pandas if Arrow fails (e.g., nested lists)
            self.logger.warning(f"Arrow CSV write failed, falling back to pandas: {e}")
            df.to_csv(filepath, index=False)
            self.logger.info(f"Saved `{len(df)}` rows to CSV via pandas: {filepath}")


# ---------------========== Handler Factory ==========--------------- #

class HandlerFactory:
    """
    Factory class for creating file handlers based on file format.
    """

    _handlers = {
        '.csv': CsvHandler,
    }

    _aliases = {
        'csv': CsvHandler,
    }

    @classmethod
    def get_handler(
        cls, filepath: Union[str, Path], logger: Logger
    ) -> BaseFileHandler:
        """
        Select an appropriate handler based on file extension.

        Args:
            filepath:   File path.
            logger: Logger instance.
        --------
        Returns: 
            Instantiated handler.
        --------
        """
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()
        handler_class = cls._handlers.get(suffix)

        if not handler_class:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: {list(cls._handlers.keys())}"
            )

        return handler_class(logger)

    @classmethod
    def get_handler_by_format(cls, fmt: str, logger: Logger) -> BaseFileHandler:
        """
        Select handler by format alias (e.g., 'csv').

        Args:
            fmt:    Format name.
            logger: Logger instance.
        --------
        Returns:
            Instantiated handler.
        --------
        """
        handler_class = cls._aliases.get(fmt.lower())

        if not handler_class:
            raise ValueError(
                f"Unsupported format: {fmt}. "
                f"Supported formats: {list(cls._aliases.keys())}"
            )

        return handler_class(logger)