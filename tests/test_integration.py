import unittest
from main import VMPlacementSystem, PhysicalNode, VirtualMachine

class TestVMPlacementSystem(unittest.TestCase):
    def setUp(self):
        nodes = [
            PhysicalNode("node1", 16.0, 32.0, 1000.0, 10.0, 0.15, "DC1"),
            PhysicalNode("node2", 24.0, 48.0, 2000.0, 15.0, 0.18, "DC2")
        ]
        self.system = VMPlacementSystem(nodes)
    
    def test_vm_request_addition(self):
        vm = VirtualMachine("vm1", 4.0, 8.0, 100.0, 2.0, 1, "user1")
        result = self.system.add_vm_request(vm)
        self.assertTrue(result)
        self.assertIn("vm1", self.system.vms)
    
    def test_optimization_and_blockchain(self):
        vm = VirtualMachine("vm1", 4.0, 8.0, 100.0, 2.0, 1, "user1")
        self.system.add_vm_request(vm)
        
        placement = self.system.optimize_placement()
        self.assertIsInstance(placement, dict)
        
        block = self.system.mine_block()
        self.assertIsNotNone(block)
        
        stats = self.system.get_system_stats()
        self.assertEqual(stats['total_vms'], 1)
        self.assertEqual(stats['placed_vms'], 1)

if __name__ == '__main__':
    unittest.main()