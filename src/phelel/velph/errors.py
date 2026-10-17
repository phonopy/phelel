"""Errors that the velph commands show to the user.

The modules in config/, jobs/ and utils/ do not use click.  They raise
VelphError, or VelphConfigError of config/io.py for an error in a velph.toml
or a velph init template.  The root group of the velph commands
(cli/velph_cmd_root.py) shows VelphError as click.ClickException: the message
is printed after "Error: ", and the exit code is 1.

"""


class VelphError(Exception):
    """Error that the velph commands show to the user by its message."""
