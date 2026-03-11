# User Configs Directory

## Overview
This directory is mounted inside the R2R Docker container and is intended for custom configuration files. Any files placed here will be accessible to the application running in the container.

## Usage
1. Place your custom configuration files in this directory.
2. Set the `R2R_CONFIG_PATH` in the `r2r.env` or `r2r-full.env` files.
3. The path format inside the container is: `/app/user_configs/<config>.toml`

## Configuration
The application uses the environment variable you set to locate your configuration file:
```
R2R_CONFIG_PATH=/app/user_configs/<config>.toml
```

For local file storage in a single-node or shared-volume deployment, you can use:

```toml
[file]
provider = "local"
base_path = "/app/storage/files"
```

Make sure `LOCAL_FILE_STORAGE_PATH` points at the same mounted path and that your Docker setup persists `/app/storage`.

If you want to use a different filename, update the `R2R_CONFIG_PATH` variable in your environment file to point to your custom file, for example:
```
R2R_CONFIG_PATH=/app/user_configs/my_custom_config.toml
```

## Troubleshooting
If you encounter configuration errors, check:
1. Your configuration file exists in this directory
2. The filename matches what's specified in `R2R_CONFIG_PATH`
3. The file has proper permissions (readable)
4. The file contains valid TOML syntax

For more detailed configuration information, see the main documentation.
