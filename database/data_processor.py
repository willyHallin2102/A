"""
    database/data_processor.py
"""
import orjson

import numpy as np
import pandas as pd

from typing import Any, Dict, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from logs.logger import Logger, LogLevel


# ---------------========== Data Columns ==========---------------- #

class DataProcessor:
    """
    Handles structured transformation of input tabular data 
    (e.g., DataFrame chunks) into consistent numpy arrays according 
    to a predefined schema.

    Each schema entry defines:
        - Expected dtype
        - Whether the column represents a stacked/nested structure 
          (e.g., JSON arrays)
    
    The processor handles malformed data gracefully and supports 
    merging of chunk-level results.
    """

    SCHEMA = {
        "dvec"      : (np.float32,  True ),
        "rx_type"   : (np.uint8,    False),
        "link_state": (np.uint8,    False),
        "los_pl"    : (np.float32,  False),
        "los_ang"   : (np.float32,  True ),
        "los_dly"   : (np.float32,  False),
        "nlos_pl"   : (np.float32,  True ),
        "nlos_ang"  : (np.float32,  True ),
        "nlos_dly"  : (np.float32,  True )
    }

    def __init__(self, logger: Logger):
        """Initialize processor with a logger instance."""
        self.logger = logger

    
    # --------------- Chunk-Level Processing --------------- #

    def process_chunk(self, chunk: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Transform a DataFrame chunk into a dictionary of typed NumPy arrays.

        Handles conversion of nested JSON-like fields and enforces 
        schema-defined dtypes for all present columns. Falls back to 
        `object` arrays for ragged or malformed data.

        Args:
            chunk (pd.DataFrame): Input data chunk to process.

        Returns:
            Dict[str, np.ndarray]: Processed arrays keyed by column name.
        """
        processed: Dict[str, np.ndarray] = {}
        for column, (dtype, need_stack) in self.SCHEMA.items():
            if column not in chunk.columns: continue

            values = chunk[column].values
            if len(values) == 0:
                processed[column] = np.empty((0,), dtype=dtype)
                continue
            
            # Handle stack/nested columns
            if need_stack and values.dtype == object:
                if isinstance(values[0], str):
                    try:
                        if len(values) > 100_000:
                            # Parallel decode for very large chunks
                            with ThreadPoolExecutor() as tpe:
                                decoded = list(tpe.map(orjson.loads, values))
                        else:
                            decoded = [orjson.loads(value) for value in values]
                        stacked = np.array(decoded, dtype=dtype)
                    
                    except Exception:
                        self.logger.warning(
                            f"Column `{column}` contains malformed JSON; coercing to object."
                        )
                        stacked = np.array([
                            self._safe_parse(value) for value in values
                        ], dtype=object)
                else:
                    try: stacked = np.array(values.tolist(), dtype=dtype)
                    except ValueError:
                        stacked = np.array(values, dtype=object)
                        self.logger.debug(
                            f"Column `{column}` stored as object due to ragged shapes"
                        )
                processed[column] = stacked
            else:
                processed[column] = values.astype(dtype, copy=False)

        return processed


    


    def concatenate_results(self, 
        results: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """
            Merge multiple chunk-level results into a single dictionary or
            arrays. Handles ragged or irregular shapes relatively gracefully
            by falling back on a simplistic `object` arrays when necessary,
            for any reason failing.

            Args:
                results:    List of chunk-level results dicts to merge 
                            together.
            --------
            Returns:
                 Concatenated dictionary of string columns and NumPy 
                 arrays, consists of all the merged results arguments. \\
        """
        if not results: return {
            key: np.empty((0,), dtype=dtype) for key, (dtype,_) in self.SCHEMA.items()
        }

        concatenated: Dict[str, np.ndarray]={}
        keys = results[0].keys()
        for key in keys:
            arrays = [result[key] for result in results if key in result]
            if not arrays: continue
            try: 
                concatenated[key] = np.concatenate(arrays, axis=0)
            except ValueError:
                # Ragged fallback
                concatenated[key] = np.array(
                    sum((array.tolist() for array in arrays), []), dtype=object
                )

        return concatenated
    

    @staticmethod
    def _safe_parse(value: Any) -> Any:
        """
            Attempt to parse a JSON-like string safely using `orjson`.
            Returns `None` on any parsing failure.

            Args:
                value: Input value (expected JSON string).

            Returns:
                Any: Parsed object or None if parsing fails.
        """
        try: return orjson.loads(value)
        except Exception: return None
