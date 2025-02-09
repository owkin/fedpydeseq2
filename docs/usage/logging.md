# Logging

## Overview

Instructions on how to configure logging using JSON and INI files. The logging configuration is divided into two parts:
1. A JSON file that specifies the path to the INI file and additional logging options.
2. An INI file that defines the logger, handler, and formatter configurations.

## JSON Configuration

The JSON configuration file contains the following keys:

- `logger_configuration_ini_path`: The path to the INI file that contains the logger configuration.
- `generate_workflow`: A dictionary with the following keys:
  - `create_workflow`: A boolean indicating whether to create a workflow.
  - `workflow_file_path`: The path to the workflow file.
  - `clean_workflow_file`: A boolean indicating whether to clean the workflow file.
- `log_shared_state_adata_content`: A boolean indicating whether to log the shared state adata content.
- `log_shared_state_size`: A boolean indicating whether to log the shared state size.

### Example JSON Configuration

```json
{
    "logger_configuration_ini_path": "/path/to/logger_configuration_template.ini",
    "generate_workflow": {
        "create_workflow": false,
        "workflow_file_path": "/path/to/workflow.txt",
        "clean_workflow_file": true
    },
    "log_shared_state_adata_content": false,
    "log_shared_state_size": false
}
```

## INI Configuration

The INI configuration file defines the loggers, handlers, and formatters. Below is an example of the INI file format:

### Example INI Configuration

```ini
; Description: This is a template for the configuration of the logger

; Define the loggers
[loggers]
keys=root

; Define the handlers
[handlers]
keys=consoleHandler

; Define the formatters
[formatters]
keys=sampleFormatter

; Root logger configuration
[logger_root]
level=WARNING
handlers=consoleHandler

; Console handler configuration
[handler_consoleHandler]
class=StreamHandler
level=WARNING
formatter=sampleFormatter
args=(sys.stdout,)

; Sample formatter configuration
[formatter_sampleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

## Purpose

### Workflow
A workflow is simply a text file that logs the different shared states during subprocess mode execution. This feature is deactivated in other Substra backends. The workflow file helps in tracking the sequence of shared states and their transitions.

### log_shared_state_adata_content
The log_shared_state_adata_content flag determines whether to log the shared state keys and local adata keys to a file. This can be useful for debugging and tracking the data being processed.

### log_shared_state_size
The log_shared_state_size flag determines whether to evaluate the size of the shared state. This is used for memory management purposes, helping to ensure that the application does not exceed memory limits.

## Usage

1. Create a JSON configuration file with the required keys and values.
2. Create an INI configuration file with the logger, handler, and formatter definitions.
3. Ensure that the `logger_configuration_ini_path` in the JSON file points to the correct INI file path.
4. Use the JSON configuration file to initialize the logging configuration in your application.

By following these steps, you can configure logging in your application using JSON and INI files.
