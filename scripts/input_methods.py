import MDAnalysis as mda
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split


def dcdtrain_writecsv(ice_file, water_file, pdb_file, fname_inp, fname_targ, nearest_neigh=10, added_vac=30):
    """
    Method to convert a dcd into a csv for training
    :param nearest_neigh: number of nearest neighbours
    :param ice_file: ice dcd trajectory e.g. 'TrajectoryIce.dcd'
    :param water_file: water dcd trajectory e.g. 'TrajectoryWater.dcd'
    :param pdb_file: structure file .e.g 'IceWater.pdb'
    :param fname_inp: CSV filename for coordinates output .e.g 'input_ice_water.csv'
    :param fname_targ: CSV filename for targets output .e.g 'targets_ice_water.csv'
    :param added_vac: amount of vacuum to add to largest dimension to generate an interface (default is 30 Angstroms)
    :return:
    """

    pdb = pdb_file
    water_dcd = water_file
    ice_dcd = ice_file

    filenames = [water_dcd, ice_dcd]
    phase_name = ["Water", "Ice"]
    for phase in range(0, 2):
        # Load simulation results with a single line
        u = mda.Universe(pdb, filenames[phase])
        nframes = len(u.trajectory)

        pbc = np.array(u.universe.dimensions[0:3])
        id_max = np.where(pbc == pbc.max())
        pbc_box = np.array([u.universe.dimensions, u.universe.dimensions])
        pbc_box[0,id_max] = pbc_box[0,id_max] + added_vac # expand one dimension by added_vac to create interface

        n = u.atoms.n_atoms

        print(phase_name[phase], " Number of frames: ", nframes, " Number atoms: ", n, " Number data rows: ", n * nframes)
        print(phase_name[phase], " PBC: ", pbc_box[0])

        dist = np.zeros(n * n).reshape(n, n)
        input_net = np.zeros(n * nframes * 3 * nearest_neigh).reshape(n * nframes, nearest_neigh * 3)

        global_count = 0
        for k, ts in enumerate(u.trajectory):  # k is frame num

            # Calculate distance between all pairs of atoms using mdanalysis function
            # d is N*(N-1)/2 length vector where N = number of atoms
            d = mda.lib.distances.self_distance_array(ts.positions, box=pbc_box[0])

            # convert d into N*N Matrix. Main diagonal elements are all 0 (distance between atom i and atom i)
            kk = 0
            for i in range(n):
                for j in range(i + 1, n):
                    dist[i, j] = d[kk]
                    dist[j, i] = d[kk]
                    kk += 1

            for i in range(n):
                # This gives the index of nearest_neigh+1 smallest values
                p = np.argpartition(dist[i], nearest_neigh+1)[:nearest_neigh+1]
                p = p[p != i]  # delete any values == 0. So now have nearest_neigh elements. (+1 was central atom)
                n_p = len(p)

                for neigh in range(n_p):
                    # calculate Delta x,y,z for atom i from neighbours.
                    # Apply pbc conditions
                    diff = ts.positions[p[neigh]] - ts.positions[i]
                    diff = diff - (pbc * np.around(diff / pbc))
                    # updated vector shape so all neighbours xyz in 1D vector
                    input_net[global_count, (neigh * 3):(neigh * 3 + 3)] = diff

                global_count += 1

        if phase == 0:
            water = np.zeros(n * nframes * 3 * nearest_neigh).reshape(n * nframes, nearest_neigh * 3)
            water[0:n*nframes] = input_net

        elif phase == 1:
            ice = np.zeros(n * nframes * 3 * nearest_neigh).reshape(n * nframes, nearest_neigh * 3)
            ice[0:n*nframes] = input_net

    # Prepare input and write to data file:
    ones = np.ones(len(water))
    zeros = np.zeros(len(water))
    y_w = np.column_stack((ones, zeros))

    ones = np.ones(len(ice))
    zeros = np.zeros(len(ice))
    y_i = np.column_stack((zeros, ones))

    input_cat = np.concatenate((water, ice), axis=0)
    y_cat = np.concatenate((y_w, y_i), axis=0)

    # Write to csv data files
    df = pd.DataFrame({"water": y_cat[:,0], "ice": y_cat[:,1]})
    df.to_csv(fname_targ,index=False,header=['water', 'ice'])

    df = pd.DataFrame(input_cat)
    df.to_csv(fname_inp, index=False)


def import_input_targ_csv(fname_inp, fname_targ, nn):
    """
    In the following a nearest neigh csv and targets are inputed for training or evaluation
    :param fname_inp: CSV structure (molecular coordinates) input file
    :param fname_targ: CSV target (molecular phase) input file
    :param nn: number of nearest neighbours
    :return: x and y for training
    """
    nn = int(nn)
    targets_y = pd.read_csv(fname_targ)
    input_x = pd.read_csv(fname_inp)

    input_x_nn = input_x.iloc[:,0:nn*3]

    return input_x_nn.values, targets_y.values


def import_input_csv(fname_inp, nn):
    """
    Read in csv x input and return values
    :param fname_inp: CSV structure (molecular coordinates) input file
    :param nn: nearest neighbours
    :return: input array
    """
    nn = int(nn)
    input_x = pd.read_csv(fname_inp)
    input_x_nn = input_x.iloc[:, 0:nn * 3]  # nn is number of nearest neighbours want
    return input_x_nn.values


def slab_import_writecsv(slab_file, pdb_file, fname_out, nearest_neigh):
    """
    Import dcd slab and save to csv and return array
    :param slab_file: slab dcd trajectory
    :param pdb_file: structure file
    :param fname_out: CSV filename for coordinates output
    :param nearest_neigh:
    :return: input array
    """

    u = mda.Universe(pdb_file, slab_file)  # Load simulation results
    nframes = len(u.trajectory)
    pbc = u.universe.dimensions[0:3]
    n = u.atoms.n_atoms

    print(" Number of frames: ", nframes, " Number atoms: ", n, " Number data rows: ", n * nframes)
    print(" PBC: ", pbc)

    dist = np.zeros(n * n).reshape(n, n)
    input_net = np.zeros(n * nframes * 3 * nearest_neigh).reshape(n * nframes, nearest_neigh * 3)

    global_count = 0
    for k, ts in enumerate(u.trajectory):  # k is frame num.

        # Calculate distance (with pbc and taking into account xyz) between all
        # pairs of atoms using mdanalysis function
        # d is N*(N-1)/2 length vector

        d = mda.lib.distances.self_distance_array(ts.positions, box=u.universe.dimensions)

        # convert d into N*N Matrix. Main diagonal elements are all 0 as dist between i-i.
        kk = 0
        for i in range(n):
            for j in range(i + 1, n):

                dist[i, j] = d[kk]
                dist[j, i] = d[kk]
                kk += 1

        for i in range(n):

            # This gives the index of nearest_neigh+1 smallest values
            p = np.argpartition(dist[i], nearest_neigh+1)[:nearest_neigh+1]
            p = p[p != i]  # Remove central atom so now have nearest_neigh elements

            n_p = len(p)

            for neigh in range(n_p):
                # calculate Delta x,y,z for atom i from neighbours.
                # Apply pbc conditions
                diff = ts.positions[p[neigh]] - ts.positions[i]
                diff = diff - (pbc * np.around(diff / pbc))
                # updated vector shape so all 10 neighbours xyz in 1D vector
                input_net[global_count, (neigh * 3):(neigh * 3 + 3)] = diff
                # input_net[global_count,neigh] = diff

            global_count += 1

    # Write input neural network input to csv data files using pandas
    df = pd.DataFrame(input_net)
    df.to_csv(fname_out, index=False)
    return input_net


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.

    from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    """

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def rotate_inputs(inputs, targets, nn, n_loops=1):
    """
    Rotate input randomly
    :param inputs: coordinate array
    :param targets: target array
    :param nn: number of nearest neighbours
    :param n_loops: number of times to loop through data
    :return: array of rotated inputs and corresponding targets
    """
    n_inputs = len(targets)

    inputs = inputs.reshape(n_inputs, nn, 3)

    new_inputs = np.zeros((n_inputs*n_loops, nn, 3))
    new_targets = np.zeros((n_inputs*n_loops, 2))


    n_rotations = 5000
    n_batch = float(int(n_inputs/n_rotations))

    for loop in range(n_loops):
        print("loop:", loop)
        id_first = loop*n_inputs
        id_last = id_first + n_inputs

        # randomise order:
        inputs, _, targets, _ = train_test_split(inputs, targets, test_size=0.0, random_state=0)
        new_targets[id_first:id_last] = targets

        random_var = []
        for bb in range(int(n_batch)):
            random_var.append(np.random.uniform(size=(3,)))

        for i in range(n_inputs):
            # ignore randnums if want no batches
            random_rot_matrix = rand_rotation_matrix(deflection=1.0) # randnums=random_var[batch_id])
            new_inputs[i+id_first] = inputs[i].dot(random_rot_matrix.transpose())

    new_inputs = new_inputs.reshape(n_inputs*n_loops,nn*3)

    return new_inputs, new_targets


def write_rotation_csv(inputs, targets, fname_inp, fname_targ):
    """
    Save rotated data set to csv
    :param inputs: input array
    :param targets: target array
    :param fname_inp: input CSV filename
    :param fname_targ: target CSV filename
    """
    df = pd.DataFrame()
    df["water"] = targets[:,0]
    df["ice"] = targets[:,1]
    df.to_csv(fname_targ,index=False)
    df = pd.DataFrame(inputs)
    df.to_csv(fname_inp, index=False)




