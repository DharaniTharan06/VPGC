import unittest
from main import GameTheoryOptimizer, PhysicalNode, VirtualMachine

class TestGameTheoryOptimizer(unittest.TestCase):
    def setUp(self):
        self.nodes = [
            PhysicalNode("node1", 16.0, 32.0, 1000.0, 10.0, 0.15, "DC1"),
            PhysicalNode("node2", 24.0, 48.0, 2000.0, 15.0, 0.18, "DC2")
        ]
        self.optimizer = GameTheoryOptimizer(self.nodes)
        self.vm = VirtualMachine("vm1", 4.0, 8.0, 100.0, 2.0, 1, "user1")
    
    def test_utility_calculation(self):
        utility = self.optimizer.calculate_utility(self.vm, self.nodes[0])
        self.assertIsInstance(utility, float)
        self.assertGreater(utility, 0)
    
    def test_resource_utilization(self):
        util = self.optimizer._calculate_resource_utilization(self.vm, self.nodes[0])
        self.assertGreaterEqual(util, 0)
        self.assertLessEqual(util, 1)
    
    def test_nash_equilibrium_placement(self):
        vms = [
            VirtualMachine("vm1", 4.0, 8.0, 100.0, 2.0, 1, "user1"),
            VirtualMachine("vm2", 6.0, 12.0, 200.0, 3.0, 2, "user2")
        ]
        placement = self.optimizer.nash_equilibrium_placement(vms)
        self.assertIsInstance(placement, dict)
        self.assertEqual(len(placement), 2)

if __name__ == '__main__':
    unittest.main()