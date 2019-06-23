import logging
import sys

class COMMON(object):
    def log_debug(self, msg):
        logging.debug(msg)
    def log_info(self, msg):
        logging.info(msg)
    def log_warning(self, msg):
        logging.warning(msg)
    def log_error(self, msg):
        logging.error(msg)
        sys.exit(-1)
