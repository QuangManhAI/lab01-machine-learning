import logging
from pathlib import Path


ERROR_LOG = Path("ERRORS.log")
PIPELINE_LOG = Path("PIPELINE.log")


def setup_error_logging():
    logger = logging.getLogger()
    if not any(getattr(handler, "baseFilename", None) == str(PIPELINE_LOG.resolve()) for handler in logger.handlers):
        handler = logging.FileHandler(PIPELINE_LOG)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    if not any(getattr(handler, "baseFilename", None) == str(ERROR_LOG.resolve()) for handler in logger.handlers):
        handler = logging.FileHandler(ERROR_LOG)
        handler.setLevel(logging.ERROR)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def run_logged(func):
    setup_error_logging()
    try:
        func()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            logging.getLogger(__name__).error(str(exc))
        raise
    except Exception:
        logging.getLogger(__name__).exception("Program failed")
        raise
