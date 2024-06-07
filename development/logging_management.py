
import logging
log_format = "%(asctime)s - %(levelname)s - %(module)s - %(lineno)d - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(format=log_format, datefmt=date_format, level=logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(filename="app.log", level=logging.DEBUG)


# logging.debug("This is a debug message")
# logging.info("Successful event")
# logging.warning("Low disk space: Only 5% free.")
# logging.error("File not found error")
# logging.critical("Database connection failed")


### Creating a Logger Object ###
show_logs = False

if show_logs:
    import logging
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)
    logger.debug("This is a debug message (won't be displayed)")
    logger.info("This is an info message (will be displayed)")


### Logging Exceptions ###
import logging
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - $(name)s - %(levelaname)s - %(message)s')

try:
    result = 10 / 0

except Exception as e:
    logging.error("An exception occurred: %s", str(e), exc_info=True)
