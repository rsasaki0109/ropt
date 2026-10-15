import unittest
from information_precision import energy_weighted_residual, column_energy, precise_information


class PrecisionTests(unittest.TestCase):
    def test_exact_binary_stiff_weak_direction_keeps_original_energy_and_gradient(self):
        # Eigenvalues 4 and 2**54+4; every original entry is exactly representable.
        a=2.**53+4.;b=2.**53
        info=(a,b,0.,a,0.,1.)
        self.assertIsNotNone(precise_information(info))
        energy,w=energy_weighted_residual((1.,-1.,0.),info)
        self.assertEqual(energy,4.)
        self.assertEqual(w,(4.,-4.,0.))
        self.assertEqual(column_energy((1.,-1.,0.),info),8.)

    def test_third_pivot_and_recorded_information_use_original_quadratic_form(self):
        a = 2.**53 + 4.
        b = 2.**53
        info = (1., 0., 0., a, b, a)
        self.assertEqual(energy_weighted_residual((0., 1., -1.), info), (4., (0., 4., -4.)))
        info = (2693538350855.096, -157146640359.0913, 0., 9168262482.076782, 0., 636.440966)
        r = (-0.0008459986500432724, -0.01450065877031801, -0.00031608221007084497)
        self.assertEqual(energy_weighted_residual(r, info)[0], 0.0012039340794135457)

    def test_synthetic_normal_information_keeps_historical_float_operations(self):
        info=(400.,30.,10.,500.,-15.,2500.);r=(.3,-.2,.01)
        self.assertIsNone(precise_information(info))
        a,b,c,d,e,f=info;x,y,z=r
        w=(a*x+b*y+c*z,b*x+d*y+e*z,c*x+e*y+f*z)
        self.assertEqual(energy_weighted_residual(r,info),(.5*sum(v*u for v,u in zip(r,w)),w))


if __name__=='__main__':unittest.main()
