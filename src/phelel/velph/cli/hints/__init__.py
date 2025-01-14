"""velph command line tool / velph-hints."""

import click

from phelel.velph.cli import cmd_root


#
# velph hints
#
@cmd_root.command("hints")
def cmd_hints():
    """Show velph command hints."""
    _show_hints("velph.toml")


def _show_hints(toml_filename: str):
    """Show hints."""
    click.echo(
        "------------------------------- velph hints -------------------------------"
    )
    click.echo('To see detailed options, "velph ... --help".')
    click.echo("")
    click.echo("# Initialization")
    click.echo('1. "velph init --help": List velph-init options.')
    click.echo(f'2. "velph init POSCAR_TYPE_FILE FOLDER": Generate "{toml_filename}".')
    click.echo(
        '3. "velph generate": Generate standardized unit cell and primitive cell.'
    )
    click.echo("4. Confirm if space-group-type is the expected one.")
    click.echo('5. Otherwise restart from step 1 using "--tolerance" option.')
    click.echo("")
    click.echo("# Unit cell relaxation (optional)")
    click.echo(f'1. Modify [vasp.relax] section in "{toml_filename}".')
    click.echo('2. "velph relax generate": Generate input files.')
    click.echo("3. Run VASP.")
    click.echo(
        '4. Restart from # Initialization using "CONTCAR" then skip '
        "# Unit cell relaxation."
    )
    click.echo()
    click.echo("# Electronic band structure calculation")
    click.echo(f'1. Modify [vasp.el_bands] sections in "{toml_filename}".')
    click.echo(
        '2. "velph el_bands generate": Generate input files for electronic band '
        "structure."
    )
    click.echo("3. Run VASP in el_bands directories.")
    click.echo(
        '4. "velph el_bands plot --window -3 4" to plot the bands from Ef-3 to Ef+4'
    )
    click.echo()
    click.echo("# NAC calculation (optional)")
    click.echo(f'1. Modify [vasp.nac] section in "{toml_filename}".')
    click.echo('2. "velph nac generate": Generate input files.')
    click.echo("3. Run VASP.")
    click.echo()
    click.echo("# Supercell calculation")
    click.echo(f'1. Modify [vasp.phelel] section in "{toml_filename}".')
    click.echo('2. "velph phelel init": Prepare finite displacement calculation. ')
    click.echo("   NAC parameters are read when NAC calculation is available.")
    click.echo('3. "velph phelel generate": Generate input files.')
    click.echo("4. Run VASP.")
    click.echo()
    click.echo("# Phonon band structure calculation")
    click.echo("Result of supercell calculation is necessary.")
    click.echo(f'1. Modify [vasp.ph_bands] sections in "{toml_filename}".')
    click.echo(
        '2. "velph ph_bands generate": Generate input files for phononic band '
        "structure."
    )
    click.echo("3. Run VASP in ph_bands directories.")
    click.echo('4. "velph ph_bands plot" to plot the phonon bands')
    click.echo()
    click.echo("# Electron transport calculation")
    click.echo(f'1. Write [vasp.transport] section in "{toml_filename}".')
    click.echo('2. (optioanl) "velph transport generate": Dry-run to find FFT-mesh.')
    click.echo("3. (optioanl) Run VASP.")
    click.echo('4. (optioanl) "velph transport check-fft": Check FFT grid.')
    click.echo('5. (optioanl) Modify "fft_mesh" in [phelel] section manually.')
    click.echo('6. "velph supercell differentiate": Generate derivatives hdf5 file')
    click.echo('7. "velph transport generate": Generate input files')
    click.echo()
    click.echo("# Electron self-energy calculation")
    click.echo(f'1. Modify [vasp.selfenergy] section in "{toml_filename}".')
    click.echo(
        '2. (optional) "velph selfenergy generate -d": Dry-run to find FFT-mesh.'
    )
    click.echo("3. (optional) Run VASP.")
    click.echo('4. (optioanl) "velph selfenergy check-fft": Check FFT grid.')
    click.echo('5. (optioanl) Modify "fft_mesh" in [phelel] section manually.')
    click.echo('6. "velph phelel differentiate": Generate derivatives hdf5 file')
    click.echo('7. "velph selfenergy generate": Generate input files')
    click.echo()
    click.echo("# Different supercell size for phonon")
    click.echo('1. Write "supercell_dimension" in [phonopy] section and')
    click.echo("2. Write [vasp.phelel.phonon.*] entry.")
    click.echo(
        "------------------------------- velph hints -------------------------------"
    )
