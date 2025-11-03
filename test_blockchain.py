#!/usr/bin/env python3

from main import VMPlacementSystem, PhysicalNode, VirtualMachine
import time

def test_blockchain_placement():
    print("=" * 70)
    print("BLOCKCHAIN-BASED VM PLACEMENT TEST")
    print("=" * 70)
    
    nodes = [
        PhysicalNode("node_1", 16.0, 32.0, 1000.0, 10.0, 0.15, "DC1"),
        PhysicalNode("node_2", 24.0, 48.0, 2000.0, 15.0, 0.18, "DC1"),
        PhysicalNode("node_3", 32.0, 64.0, 3000.0, 20.0, 0.20, "DC2"),
    ]
    
    vms_batch1 = [
        VirtualMachine("vm_1", 4.0, 8.0, 100.0, 2.0, 5, "user_a"),
        VirtualMachine("vm_2", 6.0, 12.0, 200.0, 3.0, 3, "user_b"),
    ]
    
    system = VMPlacementSystem(nodes)
    
    print("\n[TEST 1] First placement - should be accepted (no previous best)")
    print("-" * 70)
    for vm in vms_batch1:
        system.add_vm_request(vm)
    
    placement1 = system.optimize_placement()
    block1 = system.mine_block()
    
    if block1:
        print(f"\n✓ Block {block1.block_id} mined successfully")
        if block1.placement_decision:
            print(f"  Efficiency: {block1.placement_decision.overall_efficiency:.4f}")
    
    time.sleep(1)
    
    print("\n" + "=" * 70)
    print("\n[TEST 2] Second placement - adding more VMs")
    print("-" * 70)
    
    vms_batch2 = [
        VirtualMachine("vm_3", 8.0, 16.0, 300.0, 4.0, 4, "user_c"),
    ]
    
    for vm in vms_batch2:
        system.add_vm_request(vm)
    
    placement2 = system.optimize_placement()
    block2 = system.mine_block()
    
    if block2:
        print(f"\n✓ Block {block2.block_id} mined successfully")
        if block2.placement_decision:
            print(f"  Efficiency: {block2.placement_decision.overall_efficiency:.4f}")
    
    time.sleep(1)
    
    print("\n" + "=" * 70)
    print("\n[TEST 3] Third placement - should compare with previous best")
    print("-" * 70)
    
    # Add another VM
    vms_batch3 = [
        VirtualMachine("vm_4", 2.0, 4.0, 50.0, 1.0, 2, "user_d"),
    ]
    
    for vm in vms_batch3:
        system.add_vm_request(vm)
    
    placement3 = system.optimize_placement()
    block3 = system.mine_block()
    
    if block3:
        print(f"\n✓ Block {block3.block_id} mined successfully")
        if block3.placement_decision:
            print(f"  Efficiency: {block3.placement_decision.overall_efficiency:.4f}")
    
    # Display blockchain ledger
    print("\n" + "=" * 70)
    print("\n[BLOCKCHAIN LEDGER] All placement decisions:")
    print("-" * 70)
    
    best_decision = system.blockchain.get_best_placement_from_ledger()
    
    for block in system.blockchain.chain:
        if block.placement_decision:
            is_best = "⭐ BEST" if (best_decision and 
                                    block.placement_decision.decision_id == best_decision.decision_id) else ""
            print(f"\nBlock {block.block_id} {is_best}")
            print(f"  Overall Efficiency: {block.placement_decision.overall_efficiency:.4f}")
            print(f"  Resource Util: {block.placement_decision.resource_utilization:.4f}")
            print(f"  Energy Eff: {block.placement_decision.energy_efficiency:.4f}")
            print(f"  Load Balance: {block.placement_decision.load_balance_score:.4f}")
            print(f"  VMs Placed: {len(block.placement_decision.placement)}")
    
    if best_decision:
        print(f"\n{'=' * 70}")
        print(f"BEST PLACEMENT: Block {best_decision.decision_id}")
        print(f"Overall Efficiency: {best_decision.overall_efficiency:.4f}")
        print(f"{'=' * 70}")
    
    # Verify blockchain integrity
    print(f"\n[VERIFICATION] Blockchain is valid: {system.blockchain.is_valid()}")
    print(f"Total blocks: {len(system.blockchain.chain)}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_blockchain_placement()
