


### Basic Logger ###
show_basic_logger = False
if show_basic_logger:
    import logging
    logging.debug("This is a debug message")
    logging.info("This is an info message")
    logging.warning("This is a warnimng message")
    logging.error("This is an error message")
    logging.critical("This is a critical message")

    import sys
    sys.exit(0)

### Basic Configurations ###
show_basic_config = False
if show_basic_config:

    import logging
    #logging.basicConfig(level=logging.DEBUG)
    #logging.debug("This will get logged")

    logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    logging.warning('This will get logged to a file')


### Formatting the Output ###
def log_process_id():
    import logging
    logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s')
    logging.warning('This is a warning')

#log_process_id()
def log_date_time():
    import logging
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('Admin logged in')

#log_date_time()


def log_date_time_with_format():
    import logging
    logging.basicConfig(format='%(asctime) s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logging.warning('Admin logged out')

#log_date_time_with_format()



# Logging Variable Data #
def simple_log_variable():
    import logging
    name = 'John'
    logging.error('%s raised an error', name)

#simple_log_variable()


# Capturing Stack Traces
def log_stack_traces():
    import logging
    a = 5
    b = 0
    try:
        c = a / b
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)

# log_stack_traces()

def log_stack_traces_alternative():
    import logging
    a = 5
    b = 0
    try:
        c = a / b
    except Exception as e:
        logging.exception("Exception occurred")

#log_stack_traces_alternative()



### Classes and Functions ###

def create_logger_object():
    import logging
    logger = logging.getLogger('example_logger')
    logger.warning('This is a warning')

#create_logger_object()




### Using Handlers ###
def create_handlers():
    import logging
    logger = logging.getLogger(__name__)
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('file.log')
    c_handler.setLevel(logging.WARNING)
    f_handler.setLevel(logging.ERROR)
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    logger.warning('This is a warning')
    logger.error('This is an error')

#create_handlers()




### Other Configuration Methods ###
# File config.ini
simple_file_config = """
[loggers]
keys=root,sampleLogger

[handlers]
keys=consoleHandler

[formatters]
keys=sampleFormatter

[logger_root]
level=DEBUG
handlers=consoleHandler

[logger_sampleLogger]
level=DEBUG
handlers=consoleHandler
qualname=sampleLogger
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=DEBUG
formatter=sampleFormatter
args=(sys.stdout,)

[formatter_sampleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
"""

def load_config_ini_file():
    import logging
    import logging.config
    logging.config.fileConfig(fname='config.ini', disable_existing_loggers=False)

    logger = logging.getLogger(__name__)
    logger.debug('This is a debug message')



config_file_yaml = """
version: 1
formatters:
  simple:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
    formatter: simple
    stream: ext://sys.stdout
loggers:
  sampleLogger:
    level: DEBUG
    handlers: [console]
    propagate: no
root:
  level: DEBUG
  handlers: [console]
"""
def load_config_yaml_file():
    import logging
    import logging.config
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

    logger = logging.getLogger(__name__)
    logger.debug('This is a debug message')


### Additional Examples ###
show_logs = False
if show_logs:
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
show_logs = False
if show_logs:
    import logging
    logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - $(name)s - %(levelaname)s - %(message)s')

    try:
        result = 10 / 0

    except Exception as e:
        logging.error("An exception occurred: %s", str(e), exc_info=True)


