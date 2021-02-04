import logging


def configure_test_logging():
    logging.basicConfig(
        format="%(asctime)s %(levelname)-7s %(threadName)s: %(message)s",
        level=logging.INFO
    )
