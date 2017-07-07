from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import numpy as np
from numpy.linalg import norm
import os.path
import scipy.io as sio

from leelabtoolbox.stereo import io, conversion
from itertools import product

test_dir = os.path.split(__file__)[0]


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.rng_state = np.random.RandomState(seed=0)

    def test_io_browndataset(self):
        # check that results from the two are the same, as well as with raw lee version using loadmat.
        res1 = io.read_brown_image_image_database_lee(os.path.join(test_dir, 'stereo_ref', 'brown', 'V3_4.mat'))
        res2 = io.read_brown_image_image_database(os.path.join(test_dir, 'stereo_ref', 'brown', 'V3_4.bin'))
        self.assertEqual(set(res1.keys()), set(res2.keys()))

        _error_standards = {
            'range': 0.01,
            'bearing': 1e-3,
            'inclination': 1e-3,
            'intensity': 1e-6,
        }

        for key in res1:
            v1 = res1[key]
            v2 = res2[key]
            self.assertTrue(v1.shape == v2.shape)
            self.assertTrue(abs(v1 - v2).max() < _error_standards[key])

    def test_cart_and_sph_retina2(self):
        # it's unlikely that I will generate data with 0,0,0.
        numberOfCases = 20
        for i_case in range(numberOfCases):
            if i_case % 5 == 0:
                scalar_flag = True
            else:
                scalar_flag = False
            if not scalar_flag:
                # 1d to 3d
                # conversion to tuple is important.
                shape_this = tuple(self.rng_state.randint(low=1, high=200, size=self.rng_state.randint(low=1, high=4)))
            else:
                shape_this = ()
            xyz_this = self.rng_state.randn(3, *shape_this)
            if scalar_flag:
                assert xyz_this.shape == (3,)
            xyz_recompute = np.asarray(conversion.sph2cart(*conversion.cart2sph(*xyz_this, convention='retina2'),
                                                           convention='retina2'))
            self.assertEqual(xyz_this.shape, xyz_recompute.shape)
            self.assertTrue(np.allclose(xyz_this, xyz_recompute, atol=1e-8))

            dll_this = np.empty((3,) + shape_this, dtype=np.float64)
            # d
            dll_this[0] = self.rng_state.rand(*shape_this)
            # latitude
            dll_this[1] = self.rng_state.rand(*shape_this) * np.pi - np.pi / 2
            # longitude
            dll_this[2] = self.rng_state.rand(*shape_this) * 2 * np.pi - np.pi
            if scalar_flag:
                assert dll_this.shape == (3,)

            dll_recompute = np.asarray(conversion.cart2sph(*conversion.sph2cart(*dll_this, convention='retina2'),
                                                           convention='retina2'))
            self.assertEqual(dll_this.shape, dll_recompute.shape)
            self.assertTrue(np.allclose(dll_this, dll_recompute, atol=1e-8))

    def test_car_and_sph_standard(self):
        numberOfCases = 20
        for i_case in range(numberOfCases):
            if i_case % 5 == 0:
                scalar_flag = True
            else:
                scalar_flag = False
            if not scalar_flag:
                # 1d to 3d
                # conversion to tuple is important.
                shape_this = tuple(self.rng_state.randint(low=1, high=200, size=self.rng_state.randint(low=1, high=4)))
            else:
                shape_this = ()
            xyz_this = self.rng_state.randn(3, *shape_this)
            if scalar_flag:
                assert xyz_this.shape == (3,)
            xyz_recompute = np.asarray(conversion.sph2cart(*conversion.cart2sph(*xyz_this, convention='standard'),
                                                           convention='standard'))
            self.assertEqual(xyz_this.shape, xyz_recompute.shape)
            self.assertTrue(np.allclose(xyz_this, xyz_recompute, atol=1e-8))

            dll_this = np.empty((3,) + shape_this, dtype=np.float64)
            # d
            dll_this[0] = self.rng_state.rand(*shape_this)
            # latitude
            dll_this[1] = self.rng_state.rand(*shape_this) * np.pi - np.pi / 2
            # longitude
            dll_this[2] = self.rng_state.rand(*shape_this) * 2 * np.pi
            if scalar_flag:
                assert dll_this.shape == (3,)

            dll_recompute = np.asarray(conversion.cart2sph(*conversion.sph2cart(*dll_this, convention='standard'),
                                                           convention='standard'))
            self.assertEqual(dll_this.shape, dll_recompute.shape)
            self.assertTrue(np.allclose(dll_this, dll_recompute, atol=1e-8))

    def test_brown_raw_to_sph_and_xyz(self):
        # test my brown conversion routine.
        example_scene_file = os.path.join(test_dir, 'stereo_ref', 'brown', 'V3_4.mat')
        scene_struct = io.brown_raw_to_retina2_sph(io.read_brown_image_image_database_lee(example_scene_file))

        demo_struct = sio.loadmat(os.path.join(test_dir, 'stereo_ref', 'brown', 'brown_raw_to_sph_and_xyz_ref.mat'))
        # then make sure they are the same
        distance_mask = scene_struct['distance_mask']
        distance = scene_struct['distance']
        valid_mask = np.logical_not(distance_mask)
        xyz_valid = conversion.sph2cart(scene_struct['distance'][valid_mask],
                                        scene_struct['latitude'][valid_mask],
                                        scene_struct['longitude'][valid_mask], convention='retina2')
        xyz_valid = np.asarray(xyz_valid)

        # compare xyz
        xyz_3D = np.full(distance.shape + (3,), fill_value=np.nan, dtype=np.float64)
        for idx, coord_one_axis in enumerate(xyz_valid):
            xyz_3D[valid_mask, idx] = coord_one_axis
        xyz_3D_ref = demo_struct['xyzMatrixArray']
        self.assertEqual(xyz_3D.shape, xyz_3D_ref.shape)
        self.assertTrue(np.allclose(xyz_3D, xyz_3D_ref, atol=1e-8, equal_nan=True))

        # compare mask
        self.assertTrue(np.array_equal(distance_mask.ravel(order='F')[:, np.newaxis], demo_struct['noResultMask']))

        # compare shift in azimuth
        self.assertAlmostEqual(scene_struct['longitude_shift'], demo_struct['aziShiftArray'].ravel()[0], places=6)

        # compare ddp
        # first horizontal then vertical
        self.assertAlmostEqual(scene_struct['radian_per_pixel_horizontal'] * 180 / np.pi,
                               demo_struct['ddpArray'].ravel()[0], places=4)
        self.assertAlmostEqual(scene_struct['radian_per_pixel_vertical'] * 180 / np.pi,
                               demo_struct['ddpArray'].ravel()[1], places=4)
        # compare datamap
        # order in azimith, elevation, radius
        datamap = np.asarray([scene_struct['longitude'].ravel(order='F'),
                              scene_struct['latitude'].ravel(order='F'),
                              scene_struct['distance'].ravel(order='F')])
        datamap_ref = demo_struct['datamap']
        self.assertEqual(datamap.shape, datamap_ref.shape)
        self.assertTrue(np.allclose(datamap, datamap_ref, atol=1e-6))

    def test_cart2disparity_with_ref(self):
        # test with previous MATLAB implementations.
        ref_data = sio.loadmat(os.path.join(test_dir, 'stereo_ref', 'brown', 'test_cart2disparity_ref.mat'))
        image_array = ref_data['ImageArray']
        f_array = ref_data['FArray']
        results_all = ref_data['results_all']

        num_case = image_array.shape[-1]
        assert image_array.shape == (3, 1000, num_case)
        assert f_array.shape == (3, num_case)
        assert results_all.shape == (num_case, 1)

        for i_case in range(num_case):
            f_this = f_array[:, i_case]
            xyz_this = image_array[:, :, i_case]
            disparity_map_this = conversion.cart2disparity(*xyz_this, fixation_point=f_this,
                                                           infinite_fixation=False, ipd=0.065, legacy=True)[np.newaxis]
            # disparity_map_this_correct = conversion.cart2disparity(*xyz_this, fixation_point=f_this,
            #                                                infinite_fixation=False, ipd=0.065, legacy=False)[np.newaxis]
            disparity_map_this_ref = results_all[i_case, 0]
            self.assertEqual(disparity_map_this.shape, disparity_map_this_ref.shape)
            self.assertTrue(np.allclose(disparity_map_this, disparity_map_this_ref, atol=1e-6))
            # print(abs(disparity_map_this_correct-disparity_map_this_ref).max())

    def test_cart2disparity_with_special_case(self):
        # test points on VM circle (horopter) have equal disparity.
        # test that, along line of eye sight, fixation point has zero disparity, far ones have > 0 disparity
        # and near ones have < 0 disparity.
        # check <https://en.wikipedia.org/wiki/Horopter>
        # as well as 2008 Yang Liu paper, "Disparity statistics in natural scenes"
        #        by Yang Liu; Alan C. Bovik; Lawrence K. Cormack, doi:10.1167/8.11.19, especially Fig. 1
        num_case = 200
        for i_case in range(num_case):
            if i_case % 5 == 0:
                scalar_flag = True
            else:
                scalar_flag = False
            if not scalar_flag:
                # 1d to 3d
                # conversion to tuple is important.
                shape_this = tuple(self.rng_state.randint(low=1, high=20, size=self.rng_state.randint(low=1, high=4)))
            else:
                shape_this = ()

            f_this = self.rng_state.randn(3)
            ipd = self.rng_state.rand()

            # first normalize xyz to be on horopter.
            # calculate the location of horopter.
            # easy to do.
            # just by trigonometry.
            # let F at (0, 0, -z) (z>0)
            # left eye at (-b, 0, 0), right at (b, 0, 0)
            # we need to find center of VM circle (0, 0, x), such that
            # distance from eye to (0,0,x) is same as F to (0,0,x)
            # we have x^2 + b^2 = (z+x)^2.
            # so x = (b^2 - z^2)/2z.
            # this is in the normalized coordinate system, where middle of eye is at (0,0,0).
            # we need to transform it back.
            norm_f = norm(f_this)

            def get_vm_circle_center(z):
                return np.array([0, 0, ((ipd / 2) ** 2 - z ** 2) / (2 * z)], dtype=np.float64)

            vm_circle_center_normalized = get_vm_circle_center(norm_f)
            radius = abs(vm_circle_center_normalized[2] + norm_f)

            # use inverse (transpose) of rotation matrix to put vmcircle back.
            # https://math.stackexchange.com/questions/913903/inverse-of-a-rotation-matrix
            rotation_matrix = conversion._transformation_fixation(f_this,
                                                                  eye_location=np.array([0, 0, 0],
                                                                                        dtype=np.float64))

            def old_to_new(x):
                return np.matmul(x.T, rotation_matrix.T).T

            def new_to_old(x):
                return np.matmul(x.T, rotation_matrix).T

            vm_circle_center_old_coord = new_to_old(vm_circle_center_normalized)
            eye_l, eye_r = conversion._eye_locations(f_this, ipd)
            # make sure it's correct.
            # it should be equal distance to F, L, R
            d_l = norm(eye_l - vm_circle_center_old_coord)
            d_r = norm(eye_r - vm_circle_center_old_coord)
            d_f = norm(f_this - vm_circle_center_old_coord)
            # print(d_l, d_r, d_f, radius)
            assert abs(d_l - radius) < 1e-6
            assert abs(d_r - radius) < 1e-6
            assert abs(d_f - radius) < 1e-6

            # generate points that are in front of eye, and also on horoptor
            def generate_data(shape, z):
                # generate data on the circle defined by eyes and (0, 0, -z).
                shift = get_vm_circle_center(z)
                rad = norm(shift[2] + z)
                # essentially, generate one by one.
                xyz_to_use = np.empty((3,) + shape, dtype=np.float64)
                for index, _ in np.ndenumerate(np.empty(shape)):
                    ok_flag = False
                    while not ok_flag:
                        xyz_now = self.rng_state.randn(3)
                        xyz_now[[0, 2]] = xyz_now[[0, 2]] / norm(xyz_now[[0, 2]]) * rad
                        # xyz_now[1] = 0.0001
                        xyz_now = xyz_now + shift
                        if xyz_now[2] < 0:
                            ok_flag = True
                    assert ok_flag
                    if not scalar_flag:
                        xyz_to_use[(slice(None),) + index] = xyz_now
                    else:
                        return xyz_now

                return xyz_to_use

            def check_one_case(z, ref_disparity=('=', 0)):
                xyz_this_raw = generate_data(shape_this, z)
                xyz_this = new_to_old(xyz_this_raw)
                # xyz_this_normalized_2 = old_to_new(xyz_this)
                # xyz_this_normalized_shifted_2 = (xyz_this_normalized_2.T - vm_circle_center_normalized).T
                # xyz_this_normalized_shifted_2_norm_xoz = norm(xyz_this_normalized_shifted_2[[0, 2]], axis=0)
                # assert np.allclose(xyz_this_normalized_shifted_2_norm_xoz, radius, atol=1e-4)

                disparity_std, xyz_l, xyz_r = conversion.cart2disparity(*xyz_this, fixation_point=f_this,
                                                                        infinite_fixation=False,
                                                                        ipd=ipd, return_xyz=True)
                # print((xyz_l-xyz_l_rand)[0], (xyz_r-xyz_r_rand)[0])
                # print(disparity_std)
                self.assertEqual(disparity_std.shape, shape_this)
                if shape_this == ():
                    self.assertTrue(np.isscalar(disparity_std))
                # they should be all the same
                self.assertAlmostEqual(disparity_std.min(), disparity_std.max(), places=6)
                sign, dis = ref_disparity
                if sign == '=':
                    self.assertTrue(np.allclose(disparity_std, dis, atol=1e-6))
                elif sign == '>':
                    self.assertTrue(np.all(disparity_std > dis))
                else:
                    assert sign == '<'
                    self.assertTrue(np.all(disparity_std < dis))

            check_one_case(norm_f)
            check_one_case(norm_f * 1.1, ('>', 0))  # all the same. > 0
            check_one_case(norm_f * 0.9, ('<', 0))  # all the same. < 0
            # then check closer points.

    def test_cart2disparity_matmul(self):
        numberOfCases = 200

        for i_case, infinite_fix, legacy in product(range(numberOfCases), [True, False], [True, False]):
            if i_case % 5 == 0:
                scalar_flag = True
            else:
                scalar_flag = False
            if not scalar_flag:
                # 1d to 3d
                # conversion to tuple is important.
                shape_this = tuple(self.rng_state.randint(low=1, high=50, size=self.rng_state.randint(low=1, high=4)))
            else:
                shape_this = ()

            f_this = self.rng_state.randn(3)
            ipd = self.rng_state.rand()
            xyz_this = self.rng_state.randn(3, *shape_this)
            disparity_std, xyz_l, xyz_r = conversion.cart2disparity(*xyz_this, fixation_point=f_this,
                                                                    infinite_fixation=infinite_fix,
                                                                    ipd=ipd, legacy=legacy, return_xyz=True)
            disparity_std_debug, xyz_l_debug, xyz_r_debug = conversion.cart2disparity(*xyz_this, fixation_point=f_this,
                                                                                      infinite_fixation=infinite_fix,
                                                                                      debug=True, ipd=ipd,
                                                                                      legacy=legacy, return_xyz=True)
            self.assertEqual(disparity_std.shape, shape_this)
            self.assertEqual(disparity_std_debug.shape, shape_this)
            if shape_this == ():
                self.assertTrue(np.isscalar(disparity_std))
                self.assertTrue(np.isscalar(disparity_std_debug))
            # it's possible they are different, as batch matrix mul can have a little different result
            # than individually.
            # print(abs(disparity_std-disparity_std_debug).max(), disparity_std.shape)
            self.assertTrue(np.allclose(disparity_std, disparity_std_debug, atol=1e-6))

            self.assertEqual(xyz_l.shape, (3,) + shape_this)
            self.assertEqual(xyz_r.shape, (3,) + shape_this)
            self.assertEqual(xyz_l_debug.shape, (3,) + shape_this)
            self.assertEqual(xyz_r_debug.shape, (3,) + shape_this)

            self.assertTrue(np.allclose(xyz_l, xyz_l_debug, atol=1e-6))
            self.assertTrue(np.allclose(xyz_r, xyz_r_debug, atol=1e-6))

            if infinite_fix:
                self.assertTrue(np.allclose((xyz_l.T - np.asarray([ipd, 0, 0], dtype=np.float64)).T, xyz_r, atol=1e-6))
                self.assertTrue(np.allclose((xyz_l_debug.T - np.asarray([ipd, 0, 0], dtype=np.float64)).T, xyz_r_debug,
                                            atol=1e-6))

    def test_transformation_fixation_full_with_eye_moved(self):
        # old test from personal_library_suite, with stricter check.
        num_case = 200
        f_array = self.rng_state.rand(num_case, 3) * 20 - 10
        ipd_array = self.rng_state.rand(num_case)
        for i_case in range(num_case):
            this_f = f_array[i_case]
            ipd = ipd_array[i_case]
            eye_l, eye_r = conversion._eye_locations(this_f, ipd=ipd)
            rotation_l = conversion._transformation_fixation(this_f, eye_l)
            rotation_r = conversion._transformation_fixation(this_f, eye_r)
            p_l = np.matmul(rotation_l, this_f - eye_l)
            p_r = np.matmul(rotation_r, this_f - eye_r)

            self.assertEqual(p_l.shape, (3,))
            self.assertEqual(p_r.shape, (3,))
            self.assertTrue(np.allclose(p_l[:2], np.array([0, 0]), atol=1e-6))
            self.assertTrue(np.allclose(p_r[:2], np.array([0, 0]), atol=1e-6))
            self.assertLessEqual(p_l[2], 0)
            self.assertLessEqual(p_r[2], 0)

            self.assertTrue(np.allclose(p_l[2] ** 2, (ipd / 2) ** 2 + norm(this_f) ** 2, atol=1e-6))
            self.assertTrue(np.allclose(p_r[2] ** 2, (ipd / 2) ** 2 + norm(this_f) ** 2, atol=1e-6))

            # test orthogonality
            self.assertTrue(np.allclose(np.dot(eye_l - eye_r, this_f) / norm(this_f), 0, atol=1e-4))
            self.assertTrue(np.allclose(np.sum((eye_l - eye_r) ** 2), ipd ** 2, atol=1e-6))
            self.assertTrue(np.allclose(eye_l, -eye_r, atol=1e-6))

            # # make sure eyes are on correct sides. essentially disambiguating different cases satisfying ortho.
            # original
            # % make sure that eyes are on the correct sides. So that
            # % left eye is on the left side of line , and right eye is on
            # % the right side of OF, when seen from +Y to -Y, (with z rotated properly)

            this_f_proj_on_xz = this_f.copy()
            this_f_proj_on_xz[1] = 0

            # there's only one ambiguity: whether front or back, as we are only allowed to rotate y axis,
            # and eyes stay in XoZ; see actual code of _eye_locations
            # among all rotations, there are clearly only two possibilities to make OF and LR to be orthogonal.
            # (assuming F has component in all directions, and not degenerate)
            # and cross product can solve this ambiguity.
            # % use cross product to help determine the direction.
            cross_l = np.cross(eye_r, this_f_proj_on_xz)  # should give a vector pointing upwards.
            cross_r = np.cross(this_f_proj_on_xz, eye_l)

            self.assertEqual(cross_l.shape, (3,))
            self.assertEqual(cross_r.shape, (3,))

            self.assertTrue(
                np.allclose(np.dot(eye_l - eye_r, this_f_proj_on_xz) / norm(this_f_proj_on_xz), 0, atol=1e-4))
            self.assertAlmostEqual(cross_l[1], (ipd / 2) * norm(this_f_proj_on_xz), places=6)
            self.assertAlmostEqual(cross_r[1], (ipd / 2) * norm(this_f_proj_on_xz), places=6)
            self.assertTrue(np.allclose(cross_l[[0, 2]], np.array([0, 0]), atol=1e-6))
            self.assertTrue(np.allclose(cross_r[[0, 2]], np.array([0, 0]), atol=1e-6))

    def test_transformation_fixation_full(self):
        # old test from personal_library_suite
        num_case = 200
        f_array = self.rng_state.rand(num_case, 3) * 20 - 10
        ipd = 0.065
        # in practice, if we really fixate on f, then we should rotate eye_l and eye_r first as well
        # but this test still should pass without this rotation.
        eye_l = np.array([-ipd / 2.0, 0, 0], dtype=np.float64)
        eye_r = np.array([ipd / 2.0, 0, 0], dtype=np.float64)
        for i_case in range(num_case):
            this_f = f_array[i_case]
            rotation_l = conversion._transformation_fixation_legacy(this_f, eye_l)
            rotation_r = conversion._transformation_fixation_legacy(this_f, eye_r)
            p_l = np.matmul(rotation_l, this_f - eye_l)
            p_r = np.matmul(rotation_r, this_f - eye_r)

            self.assertEqual(p_l.shape, (3,))
            self.assertEqual(p_r.shape, (3,))
            self.assertTrue(np.allclose(p_l[:2], np.array([0, 0]), atol=1e-6))
            self.assertTrue(np.allclose(p_r[:2], np.array([0, 0]), atol=1e-6))
            self.assertLessEqual(p_l[2], 0)
            self.assertLessEqual(p_r[2], 0)

    def test_transformation_fixation(self):
        # old test from personal library_suite
        # numberOfCases = 200;
        # FArray = rand(3,numberOfCases)*5;
        # FArray(3,:) = -FArray(3,:); % z should be negative.
        # IODistance = 0.065;
        # L = [-IODistance/2;0;0];
        # R = [IODistance/2;0;0];
        # for iCase = 1:numberOfCases
        #     thisF = FArray(:,iCase);
        #     [RFL,TL] = stereo.transformation_fixation(thisF,L);
        #     [RFR,TR] = stereo.transformation_fixation(thisF,R);
        #
        #     PL = RFL*(thisF - TL);
        #     PR = RFR*(thisF - TR);
        #
        #     testCase.assertEqual(PL(1:2),[0;0],'AbsTol',1e-6);
        #     testCase.assertEqual(PR(1:2),[0;0],'AbsTol',1e-6);
        #     testCase.assertLessThanOrEqual(PL(3),0); % sth I forgot to check before when doing cortex_toolkit
        #     testCase.assertLessThanOrEqual(PR(3),0);
        # end
        num_case = 200
        f_array = self.rng_state.rand(num_case, 3) * 5
        f_array[:, 2] = -f_array[:, 2]
        ipd = 0.065
        # in practice, if we really fixate on f, then we should rotate eye_l and eye_r first as well
        # but this test still should pass without this rotation.
        eye_l = np.array([-ipd / 2.0, 0, 0], dtype=np.float64)
        eye_r = np.array([ipd / 2.0, 0, 0], dtype=np.float64)
        for i_case in range(num_case):
            this_f = f_array[i_case]
            rotation_l = conversion._transformation_fixation_legacy(this_f, eye_l)
            rotation_r = conversion._transformation_fixation_legacy(this_f, eye_r)
            p_l = np.matmul(rotation_l, this_f - eye_l)
            p_r = np.matmul(rotation_r, this_f - eye_r)

            self.assertEqual(p_l.shape, (3,))
            self.assertEqual(p_r.shape, (3,))
            self.assertTrue(np.allclose(p_l[:2], np.array([0, 0]), atol=1e-6))
            self.assertTrue(np.allclose(p_r[:2], np.array([0, 0]), atol=1e-6))
            self.assertLessEqual(p_l[2], 0)
            self.assertLessEqual(p_r[2], 0)

    def test_eye_loc_debug(self):
        num_case = 200
        f_array = self.rng_state.randn(num_case, 3)
        ipd_array = self.rng_state.rand(num_case)

        for i_case in range(num_case):
            this_f = f_array[i_case]
            this_ipd = ipd_array[i_case]
            l1, r1 = conversion._eye_locations(this_f, this_ipd)
            l2, r2 = conversion._eye_locations(this_f, this_ipd, debug=True)
            self.assertEqual(l1.shape, (3,))
            self.assertEqual(l2.shape, (3,))
            self.assertEqual(r1.shape, (3,))
            self.assertEqual(r2.shape, (3,))
            self.assertTrue(np.allclose(l1, l2, atol=1e-6))
            self.assertTrue(np.allclose(r1, r2, atol=1e-6))

    def test_xiong_computation(self):
        # load old data
        ref_data = sio.loadmat(os.path.join(test_dir, 'stereo_ref', 'brown', 'test_xiong_computation.mat'))
        disp_xiong = ref_data['patchDisp_reshaped_xiong']
        disp_ref = ref_data['patchDisp_reshaped']

        # then let's compute it using legacy version as well as new version.
        xyz_data = sio.loadmat(os.path.join(test_dir, 'stereo_ref', 'brown', 'V1_11_xyz.mat'))['V1_11_xyz']
        xyz_data = np.transpose(xyz_data, (2, 0, 1))
        fixation_index = (255, 1237)
        rowIndex = slice(fixation_index[0] - 100, fixation_index[0] + 101)
        colIndex = slice(fixation_index[1] - 100, fixation_index[1] + 101)
        fixationThis = xyz_data[:, fixation_index[0], fixation_index[1]]
        fixationThis = fixationThis / norm(fixationThis) * 3.9280
        patchXYZ = xyz_data[:, rowIndex, colIndex]
        # make all nan to be 1.
        mask = np.isnan(patchXYZ[0])
        assert np.array_equal(mask, np.isnan(patchXYZ[1]))
        assert np.array_equal(mask, np.isnan(patchXYZ[2]))
        patchXYZ[:, mask] = 1

        disp_legacy = conversion.cart2disparity(*patchXYZ, fixation_point=fixationThis, infinite_fixation=False,
                                                ipd=0.065,
                                                legacy=True)
        disp_new = conversion.cart2disparity(*patchXYZ, fixation_point=fixationThis, infinite_fixation=False, ipd=0.065,
                                             legacy=False)

        disp_legacy = -disp_legacy / np.pi * 180
        disp_new = -disp_new / np.pi * 180

        self.assertEqual(disp_xiong.shape, (201, 201))
        self.assertEqual(disp_ref.shape, (201, 201))
        self.assertEqual(disp_legacy.shape, (201, 201))
        self.assertEqual(disp_new.shape, (201, 201))
        valid_mask = np.logical_not(mask)

        # xiong's method (which follows 2008 Yang Liu's paper) is simply wrong.
        # check figure of test_xiong_computation.
        # his method is only right for points in same plane as eyes and lines of sight,
        # assuming that pixel per degree is really uniform.
        # for other points, his method will assume points are further than they actually are.
        # because he used raw range, instead of *projection* of range on the plane when computing x_p and z_p.

        self.assertTrue(np.allclose(disp_legacy[valid_mask], disp_ref[valid_mask], atol=1e-6))

        print('correct vs. xiong', abs(disp_new[valid_mask] - disp_xiong[valid_mask]).max())
        print('correct vs. before', abs(disp_new[valid_mask] - disp_ref[valid_mask]).max())


if __name__ == '__main__':
    unittest.main()
