import argparse
import glob

import numpy as np
from bundle_adjust import geo_utils

import utils


def utm_to_geocentric(inFile, outFile, zonestring):
    print('-------------------------')
    print('inFile: ', inFile)
    print('outFile: ', outFile)
    pts3d = np.loadtxt(inFile, dtype='float')

    easts = pts3d[:, 0]
    norths = pts3d[:, 1]
    alts = pts3d[:, 2]

    lons, lats = geo_utils.lonlat_from_utm(easts, norths, zonestring)

    x, y, z = utils.geodetic_to_ecef(lats, lons, alts)

    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    z = z[:, np.newaxis]
    pts3d = np.hstack((x, y, z))

    np.savetxt(outFile, pts3d, fmt="%lf", delimiter=' ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', type=str, required=True, help='Directory containing 3D point files')
    parser.add_argument('--aoi_id', type=str, required=True,
                        help='Area of interest ID (e.g., beijing_xxx, shanghai_xxx)')
    args = parser.parse_args()

    # Mapping of AOI prefixes to UTM zone strings
    aoi_prefix_to_zone = {
        "beijing": "50",
        "shanghai": "51",
        "wuhan": "50",
        "shenzhen": "50",
        "xian": "49",
        "nanjing": "51",
        "chengdu": "48",
        "jax": "17",
        "dji": "38"
    }

    # Extract prefix from the aoi_id
    aoi_id = args.aoi_id
    aoi_prefix = aoi_id.split('_')[0].lower()

    # Determine the UTM zone string based on the prefix
    zonestring = aoi_prefix_to_zone.get(aoi_prefix, None)

    if zonestring is None:
        raise ValueError(
            f"AOI not valid. Expected aoi_id to start with one of the specified cities (beijing, shanghai, wuhan, shenzhen, xian, nanjing, chengdu), or 'JAX' or 'Dji', but received {aoi_id}"
        )

    print("Determined UTM zone:", zonestring)
    # Find and process all 3D point files in the specified directory
    densedepth_files = sorted(glob.glob(args.file_dir + "/*_3DPts.txt"))

    for t, densedepth_file in enumerate(densedepth_files):
        print('--------------', t, densedepth_file, '--------------')
        inFile = densedepth_file
        outFile = densedepth_file[:-4] + "_ecef.txt"
        utm_to_geocentric(inFile, outFile, zonestring)
