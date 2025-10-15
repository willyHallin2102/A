"""
    logs/logger.py
    --------------
    Contains implementation of the logger instance, will construct a
    logger file where each call is being stored as well as optional 
    usage of console for feedback information during runtime.
"""
import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler

import os, sys, time
import orjson, uuid

from pathlib import Path
from datetime import datetime
from threading import Lock
from queue import Queue
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union
from enum import Enum



# ---------------========== Log Level Enum ==========--------------- #

class LogLevel(Enum):
    """
    Defines standard log levels for use throughout the system, provides
    multiple layers for various amount of feedback, including following,
    Levels (in increasing order of severity):
    - DEBUG   : Detailed diagnostic information.
    - INFO    : General operational messages.
    - WARNING : Indicates potential issues or unexpected 
                behaviour.
    - ERROR   : Serious problem preventing part of the 
                system from functioning.
    - CRITICAL: Severe error that may cause a full system 
                failure.
    -------------------------------------------
    Example:
        >>> Logger.get_logger("app").log(
                LogLevel.INFO, "Application started"
            )
    Uses `import logging` to provide the loggings.
    """
    DEBUG       : int=logging.DEBUG
    INFO        : int=logging.INFO
    WARNING     : int=logging.WARNING
    ERROR       : int=logging.ERROR
    CRITICAL    : int=logging.CRITICAL


# ---------------========== JSON Formatter ==========--------------- #

class JsonFormatter(logging.Formatter):
    """
    Json formatter implementation, manages and define the layout for how the
    json logs are formatted when being stored.
    """
    pid = os.getpid()

    def format(self, record: logging.LogRecord) -> str:
        """
            The base elements each entry logged will contain by making additions.
        """
        payload = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "timestamp": datetime.utcfromtimestamp(record.created)
                .isoformat(timespec="milliseconds") + "Z",
            "pid": self.pid,
        }

        # Include any user-supplied fields
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "process", "processName", "message"
            ): payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return orjson.dumps(payload).decode("utf-8")
    

# ---------------========== Console Formatter ==========--------------- #

class ConsoleFormatter(logging.Formatter):
    """
        Formatter handling the appearance of the console information
        to each added entry logged. For visual preemptive, different
        logging levels are coloured differently.
    """
    COLORS = {
        "DEBUG"     : "\033[96m",
        "INFO"      : "\033[94m",
        "WARNING"   : "\033[93m",
        "ERROR"     : "\033[91m",
        # "CRITICAL": "\033[41m",
        "CRITICAL"  : "\033[1;41m",
        # "CRITICAL": "\033[1;107;31m",  # Bold red text on white background
        "RESET"     : "\033[0m",
    }

    def format(self, record: logging.LogRecord):
        color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS["RESET"]
        return f"{color}{record.levelname:<8}{reset} | {record.name:<15} | {record.getMessage()}"


# ---------------========== Logger Implementation ==========--------------- #

class Logger:
    """
    Logger instance with asynchronous capability, supporting JSON 
    formatting, and console formatting outputs.

    Logs are handled through a shared `QueueListener` and 
    `QueueHandler` pair to minimize I/O contention in `threading`
    multi-threading environments.

    Features:
    ---------
        - Rotating file logs (configurable maxsize + backup count)
        - JSON log output (structured logging --disk)
        - Colorized console logging (During Runtime --no storing)
        - Context manager for timing code blocks to support measuring 
            the time complexity for various methods/functions.
    """
    _instances: Dict[str, "Logger"] = {}
    _instances_lock = Lock()
    _queue_listener: Optional[QueueListener] = None
    _shared_queue: Optional[Queue] = None
    _shared_handlers = []
    _log_filepath: Optional[Path] = None

    def __init__(self,
        name: str, level: Union[int, LogLevel]=LogLevel.INFO,
        json_format: bool=True, use_console: bool=True, 
        max_bytes: int=1024, # bytes
        backup_count: int=5
    ):
        """ --------------------------
            Initialize Logger Instance
        """
        # Constructing the directory, sign up the archives for logs as root
        self.directory = Path(__file__).parent / "logs"
        self.directory.mkdir(parents=True, exist_ok=True)

        # Parameters for the logger being defined
        self.name = name
        self.level = level.value if isinstance(level, LogLevel) else int(level)
        self.json_format, self.use_console = json_format, use_console
        self.max_bytes, self.backup_count = max_bytes, backup_count
        
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)
        self.logger.propagate = False

        # Handlers for the logger, such as sharing queue (logger).
        self._setup_handlers()
    

    def __repr__(self): return f"<Logger name={self.name} level={self.level}>"


    # --------------- Setup --------------- #

    def _setup_handlers(self):
        """

        """
        # Remove old handlers ... duplicates otherwise
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            try: handler.close()
            except Exception: pass # No error message for now
        
        if not Logger._shared_queue: Logger._shared_queue = Queue(-1)

        queue_handler = QueueHandler(Logger._shared_queue)
        queue_handler.setLevel(logging.NOTSET)
        self.logger.addHandler(queue_handler)
        
        # Initialize shared handlers if not done yet
        if not Logger._queue_listener:
            Logger._init_shared_handlers(self)
    
    
    @classmethod
    def _init_shared_handlers(cls, instance: "Logger"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        identifier = uuid.uuid4().hex[:8]

        filepath = instance.directory / f"logger-{timestamp}-{identifier}.log"
        cls._log_filepath = filepath

        file_handler = RotatingFileHandler(
            filepath, maxBytes=instance.max_bytes,
            backupCount=instance.backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(
            JsonFormatter() if instance.json_format else logging.Formatter()
        )
        file_handler.setLevel(logging.NOTSET)
        cls._shared_handlers.append(file_handler)

        if instance.use_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ConsoleFormatter())
            console_handler.setLevel(logging.NOTSET)
            cls._shared_handlers.append(console_handler)

        cls._queue_listener = QueueListener(
            cls._shared_queue, *cls._shared_handlers, respect_handler_level=True
        )

        cls._queue_listener.start()


    # -------------------- Logging API -------------------- #

    def log(self, level: Union[int, LogLevel], msg: str, **extra: Any):
        """
        Log a message with a specified severity level.

        Args:
            level:  Logging level or LogLevel enum.
            msg:    Log message
            **extra:    Arbitrary key-value pairs include in
                        JSON logs.
        """
        lvl = level.value if isinstance(level, LogLevel) else level
        self.logger.log(lvl, msg, extra=extra)

    def debug(self, msg: str, **extra):     self.log(LogLevel.DEBUG, msg, **extra)
    def info(self, msg: str, **extra):      self.log(LogLevel.INFO, msg, **extra)
    def warning(self, msg: str, **extra):   self.log(LogLevel.WARNING, msg, **extra)
    def error(self, msg: str, **extra):     self.log(LogLevel.ERROR, msg, **extra)
    def critical(self, msg: str, **extra):  self.log(LogLevel.CRITICAL, msg, **extra)
    def exception(self, msg: str, **extra): self.logger.exception(msg, extra=extra)


    # ---------------------- Context Utils ---------------------- #

    @contextmanager
    def time_block(self, label="Execution time", level=LogLevel.INFO, **extra):
        """
        Context manager to measure the execution time of a code block. 
        A flexible timer for test and log performance of snippets, in 
        particular methods/functions.

        Args:
            label:  Descriptive name for the timing log.
            level:  Logging level used to emit the timing result.
            **extra:    Additional context to include in the log
                        entry.
        --------
        Example:
            >>> with logger.time_block("Time Took")
                    func1()
                    func2()
        This block will time and perform `func1` and `func2` and log 
        the time took to compute the block. \\
        """
        start = time.perf_counter()
        try: yield
        finally:
            elapsed = time.perf_counter() - start
            self.log(level, f"{label}: {elapsed:.6f}s", **extra)


    # ---------------------- Class API ---------------------- #

    @classmethod
    def get_logger(cls, name: str, **kwargs) -> "Logger":
        """
        Get or create a named logger instance.

        Ensures a singleton logger per unique name, sharing
        a background `QueueListener` for asynchronous logging.

        Args:
            name:   Identifier for the logger.
            **kwargs:   Optional override for logger configuration.
        -----
        
        Returns:
            Configured logger instance.
        --------
        """
        with cls._instances_lock:
            if name not in cls._instances:
                cls._instances[name] = Logger(name, **kwargs)
            return cls._instances[name]

    @classmethod
    def shutdown_all(cls):
        """
        Gracefully stops all active loggers and close the 
        current active file-handlers. Stops the shared 
        `QueueListener`, closes all handlers, and clears 
        the internal logger registry.
        """
        if cls._queue_listener: cls._queue_listener.stop()
        
        for handler in list(cls._shared_handlers):
            try: handler.close()
            except: pass

        for instance in cls._instances.values():
            for handler in list(instance.logger.handlers):
                instance.logger.removeHandler(handler)
                try: handler.close()
                except: pass

        cls._instances.clear()
        cls._shared_handlers.clear()
        cls._shared_queue = None
        cls._queue_listener = None

        logging.shutdown()
