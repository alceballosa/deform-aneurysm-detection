import argparse
import logging
import os
import pandas as pd
import multiprocessing as mp
import textwrap
import datetime
import tempfile

import shutil
import json
import time

def listener_process(queue, log_file):
    setup_logging(log_file)
    while True:
        record = queue.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)

def execute_and_log(command, log_queue):
    # Create a temporary file to capture the command's output
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        # Execute the command and redirect its output to the temporary file
        status = os.system(f"{command} > {temp_file.name} 2>&1")
        
        # Read the content of the temporary file
        with open(temp_file.name, 'r') as file:
            output = file.read()
            
            # Send the content to the logging queue
            log_to_queue(log_queue, output)
    return 0 if status == 0 else status

def setup_logging(log_file):
#     log_file = os.path.join(output_path, "vessel_segmentation.log")
#     log_file = "./vessel_seg2/model_weights/aneurysmDetection/Outputs/pipeline.log"
    
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    # Also print logs to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    

def log_to_queue(log_queue, message, level=logging.INFO, name='root'):
    """
    Send a log message to the logging queue.

    Parameters:
    - log_queue (multiprocessing.Queue): The logging queue.
    - message (str): The log message.
    - level (int, optional): The logging level (e.g., logging.INFO, logging.WARNING). Defaults to logging.INFO.
    - name (str, optional): The logger name. Defaults to 'root'.
    """
    log_record = logging.LogRecord(name=name, level=level, pathname=__file__, 
                                   lineno=0, msg=message, args=(), exc_info=None)
    log_queue.put(log_record)
