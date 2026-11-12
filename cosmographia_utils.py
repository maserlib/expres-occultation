"""Helpers for generating Cosmographia catalogs and movie scripts."""

import json
from pathlib import Path

import numpy


def load_mission_config(config_path, spacecraft_name):
    """Return the mission configuration, accepting configured aliases."""
    with Path(config_path).open(encoding="utf-8") as config_file:
        missions = json.load(config_file)["missions"]

    requested_name = spacecraft_name.strip().lower()
    for key, config in missions.items():
        names = [key] + config.get("aliases", [])
        if requested_name in (name.lower() for name in names):
            return config

    supported = ", ".join(config["canonical_name"] for config in missions.values())
    raise ValueError(f"Unsupported spacecraft {spacecraft_name!r}. Supported: {supported}.")


def build_spice_dict(mission_config, cosmographia_path, year=None):
    """Build Cosmographia's SPICE catalog using paths relative to its install."""
    cosmographia_path = Path(cosmographia_path).expanduser().resolve()
    if "spice_catalog_pattern" in mission_config:
        if year is None:
            raise ValueError("A year is required to load this mission's SPICE catalog.")
        catalog_path = cosmographia_path / mission_config["spice_catalog_pattern"].format(year=year)
        if not catalog_path.is_file():
            raise FileNotFoundError(f"SPICE catalog not found: {catalog_path}")
        with catalog_path.open(encoding="utf-8") as catalog_file:
            spice_catalog = json.load(catalog_file)

        anchor = mission_config.get("spice_kernel_anchor")
        if anchor:
            relocated_kernels = []
            for kernel_path in spice_catalog.get("spiceKernels", []):
                kernel_path = Path(kernel_path)
                if kernel_path.is_absolute():
                    try:
                        anchor_index = kernel_path.parts.index(anchor)
                    except ValueError as error:
                        raise ValueError(
                            f"Kernel path does not contain the configured anchor "
                            f"{anchor!r}: {kernel_path}"
                        ) from error
                    kernel_path = Path(*kernel_path.parts[anchor_index:])
                relocated_kernels.append(str(cosmographia_path / kernel_path))
            spice_catalog["spiceKernels"] = relocated_kernels

        return spice_catalog

    return {
        "version": "1.0",
        "name": f"{mission_config['canonical_name']} kernels",
        "spiceKernels": [
            str(cosmographia_path / relative_path)
            for relative_path in mission_config["spice_kernels"]
        ],
    }


def select_frequency_indices(frequencies, mode, requested_frequency=None, count=4):
    """Select all, nearest, or regularly spaced frequency indices."""
    frequencies = numpy.asarray(frequencies)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("Frequencies must be a non-empty one-dimensional array.")

    normalized_mode = mode.strip().lower()
    if normalized_mode in ("a", "all"):
        return None
    if normalized_mode in ("n", "nearest"):
        if requested_frequency is None or not numpy.isfinite(requested_frequency):
            raise ValueError("Provide a finite requested frequency in MHz.")
        return [int(numpy.argmin(numpy.abs(frequencies - requested_frequency)))]
    if normalized_mode in ("s", "sample", "sampled"):
        if not isinstance(count, int) or not 1 <= count <= frequencies.size:
            raise ValueError(f"Frequency count must be between 1 and {frequencies.size}.")
        return numpy.rint(numpy.linspace(0, frequencies.size - 1, num=count)).astype(int)

    raise ValueError("Choose 'all', 'nearest', or 'sample'.")


def build_expres_item(start_time, end_time, source_xyz, distance_scale, center, frequency_indices=None):
    """Build the Cosmographia items for one ExPRES time step."""
    if frequency_indices is None:
        selected_xyz = source_xyz
    else:
        indices = numpy.asarray(frequency_indices, dtype=int).reshape(-1)
        selected_xyz = source_xyz[indices, :, :]

    # In the CDF data, all sources (all frequencies for each modelled magnetic field line) are present. 
    # Sources that are not visible for the modelled observer (or that are not emitting) are masked (numpy masked array)
    # We want to keep all valid data, i.e., the sources that are not masked. The `.compressed()` method on a numpy
    # masked array is flattening the array, keeping only the valid data (non masked).


    x = selected_xyz[:, :, 0].compressed()
    y = selected_xyz[:, :, 1].compressed()
    z = selected_xyz[:, :, 2].compressed()

    return [
        {
            "class": "other",
            "name": "",
            "name": f"{start_time} - {index}",
            #"name": f"{start_time} - {index} - ExPRES source",
            "trajectoryFrame": {"type": "Spice", "name": f"IAU_{center.upper()}"},
            "center": center,
            "startTime": start_time.isoformat(),
            "endTime": end_time.isoformat(),
            "geometry": {"type": "Globe", "radii": [10000, 10000, 10000]},
            "label": {"color": "e60909", "fadeSize": 1000000, "showText": "0"},
            "trajectory": {
                "type": "FixedPoint",
                "position": [x[index] * distance_scale, y[index] * distance_scale, z[index] * distance_scale],
            },
        }
        for index in range(len(x))
    ]


def build_expres_catalog(flyby_name, times, source_positions, dt, mission_config, distance_scale, frequency_indices=None):
    """Build a full ExPRES catalog from time-indexed source positions."""
    catalog = {
        "version": "1.0",
        "name": f"ExPRES Simulation for {flyby_name}",
        "items": [],
    }
    for current_time, current_sources in zip(times, source_positions):
        catalog["items"].extend(
            build_expres_item(
                current_time,
                current_time + dt,
                current_sources,
                distance_scale,
                mission_config["center"],
                frequency_indices,
            )
        )
    return catalog


def build_spacecraft_catalog(mission_config, flyby_name, start_time, end_time):
    """Build the Cosmographia spacecraft catalog."""
    spacecraft_name = mission_config["canonical_name"]
    return {
        "version": "1.0",
        "name": f"{spacecraft_name} Spacecraft ({flyby_name})",
        "items": [
            {
                "class": "spacecraft",
                "name": spacecraft_name,
                "startTime": start_time.strftime("%Y-%m-%d %H:%M:%S.%f UTC"),
                "endTime": end_time.strftime("%Y-%m-%d %H:%M:%S.%f UTC"),
                "center": mission_config["center"],
                "trajectory": {
                    "type": "Spice",
                    "target": spacecraft_name,
                    "center": mission_config["center"],
                },
                "bodyFrame": {"type": "Spice", "name": mission_config["body_frame"]},
            }
        ],
    }


def build_scenario_dict(spacecraft_name, flyby_name, required_catalogs):
    """Build a scenario referencing the required catalogs by absolute path."""
    return {
        "version": "1.0",
        "name": f"{spacecraft_name} {flyby_name} Flyby",
        "require": [str(Path(path).resolve()) for path in required_catalogs],
    }


def build_movie_script(view, spacecraft_name, center, start_time, config_path,
                       scenario_path, expres_catalog_path, movie_path, fov,
                       simulation_duration):
    """Render a Cosmographia scripting file for a POV or top-down movie."""
    if view == "pov":
        camera_commands = """\
cosmo.pointAtObject({center!r}).wait(waittime)
cosmo.setFov({fov}, waittime)
cosmo.rollToBodyFixedVector({center!r}, [0.0, 0.0, 1.0], 2).wait(waittime)
cosmo.trackObject({center!r})
""".format(center=center, fov=fov)
    elif view == "top":
        camera_commands = """\
cosmo.moveToPov({center!r}, [0, 0, 2 * 1900000.0], [0, 0, -1], [1, 0, 0], waittime)
cosmo.setCameraToInertialFrame()
cosmo.setFov({fov}, waittime)
""".format(center=center, fov=fov)
    else:
        raise ValueError("View must be 'pov' or 'top'.")

    script = """\
import cosmoscripting
cosmo = cosmoscripting.Cosmo()

waittime = 1
window_resolution = '1080p'
config_file = {config_path!r}
scenario_file = {scenario_path!r}
expres_file = {expres_catalog_path!r}

cosmo.loadCatalogFile(config_file).wait(waittime)
cosmo.loadCatalogFile(scenario_file).wait(waittime)
cosmo.loadCatalogFile(expres_file).wait(waittime)
cosmo.displayNote({note!r}, 2).wait(0)
cosmo.setWindowResolution(window_resolution)
cosmo.setVideoRecordingResolution(window_resolution)
cosmo.setTime({start_time!r})
cosmo.setTimeRate(180)
cosmo.pause()
cosmo.showToolBar()
cosmo.showStatusMessages()
cosmo.showInfoText()
cosmo.showNormalWindow()
cosmo.selectObject({center!r}).wait(waittime)
cosmo.setCentralObject({spacecraft_name!r}).wait(waittime)
cosmo.setCameraPosition([0.0, 0.0, 0.0]).wait(waittime)
{camera_commands}
cosmo.unpause()
cosmo.startRecordingVideoToFile({movie_path!r})
cosmo.waitSim({simulation_duration})
cosmo.stopRecordingVideo()
cosmo.pause()
cosmo.wait(waittime)
cosmo.quit()
""".format(
        config_path=str(config_path),
        scenario_path=str(scenario_path),
        expres_catalog_path=str(expres_catalog_path),
        note=f"Set time to {start_time}",
        start_time=start_time.strftime("%Y-%m-%d %H:%M:%S.%f UTC"),
        center=center,
        spacecraft_name=spacecraft_name,
        camera_commands=camera_commands.rstrip(),
        movie_path=str(movie_path),
        simulation_duration=simulation_duration,
    )
    return script


def write_json(path, data):
    """Write JSON, creating the parent directory when needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)
    return path
