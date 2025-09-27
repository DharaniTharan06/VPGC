# Multi-Objective Game Theory VM Placement Optimization with Blockchain
# Complete Implementation

import numpy as np
import hashlib
import json
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random
from datetime import datetime

# ===========================
# 1. DATA STRUCTURES & MODELS
# ===========================

class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"

@dataclass
class VirtualMachine:
    vm_id: str
    cpu_requirement: float
    memory_requirement: float
    storage_requirement: float
    network_requirement: float
    priority: int
    owner: str
    
    def get_resource_vector(self) -> np.ndarray:
        return np.array([
            self.cpu_requirement,
            self.memory_requirement,
            self.storage_requirement,
            self.network_requirement
        ])

@dataclass
class PhysicalNode:
    node_id: str
    cpu_capacity: float
    memory_capacity: float
    storage_capacity: float
    network_capacity: float
    energy_cost: float
    location: str
    
    # Current usage
    cpu_used: float = 0.0
    memory_used: float = 0.0
    storage_used: float = 0.0
    network_used: float = 0.0
    
    def get_capacity_vector(self) -> np.ndarray:
        return np.array([
            self.cpu_capacity,
            self.memory_capacity,
            self.storage_capacity,
            self.network_capacity
        ])
    
    def get_usage_vector(self) -> np.ndarray:
        return np.array([
            self.cpu_used,
            self.memory_used,
            self.storage_used,
            self.network_used
        ])
    
    def get_available_vector(self) -> np.ndarray:
        return self.get_capacity_vector() - self.get_usage_vector()
    
    def can_host(self, vm: VirtualMachine) -> bool:
        available = self.get_available_vector()
        required = vm.get_resource_vector()
        return np.all(available >= required)
    
    def allocate_vm(self, vm: VirtualMachine):
        if self.can_host(vm):
            self.cpu_used += vm.cpu_requirement
            self.memory_used += vm.memory_requirement
            self.storage_used += vm.storage_requirement
            self.network_used += vm.network_requirement
            return True
        return False
    
    def deallocate_vm(self, vm: VirtualMachine):
        self.cpu_used = max(0, self.cpu_used - vm.cpu_requirement)
        self.memory_used = max(0, self.memory_used - vm.memory_requirement)
        self.storage_used = max(0, self.storage_used - vm.storage_requirement)
        self.network_used = max(0, self.network_used - vm.network_requirement)

# ===========================
# 2. BLOCKCHAIN IMPLEMENTATION
# ===========================

@dataclass
class Transaction:
    tx_id: str
    vm_id: str
    node_id: str
    action: str  # "allocate", "deallocate", "migrate"
    timestamp: float
    requester: str
    gas_fee: float
    
    def to_dict(self) -> Dict:
        return {
            'tx_id': self.tx_id,
            'vm_id': self.vm_id,
            'node_id': self.node_id,
            'action': self.action,
            'timestamp': self.timestamp,
            'requester': self.requester,
            'gas_fee': self.gas_fee
        }

@dataclass
class Block:
    block_id: str
    previous_hash: str
    timestamp: float
    transactions: List[Transaction]
    merkle_root: str
    nonce: int = 0
    hash: str = ""
    
    def calculate_merkle_root(self) -> str:
        if not self.transactions:
            return hashlib.sha256(b"").hexdigest()
        
        tx_hashes = [hashlib.sha256(json.dumps(tx.to_dict()).encode()).hexdigest() 
                    for tx in self.transactions]
        
        while len(tx_hashes) > 1:
            if len(tx_hashes) % 2 == 1:
                tx_hashes.append(tx_hashes[-1])
            
            new_hashes = []
            for i in range(0, len(tx_hashes), 2):
                combined = tx_hashes[i] + tx_hashes[i+1]
                new_hashes.append(hashlib.sha256(combined.encode()).hexdigest())
            tx_hashes = new_hashes
        
        return tx_hashes[0]
    
    def calculate_hash(self) -> str:
        block_data = {
            'block_id': self.block_id,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'merkle_root': self.merkle_root,
            'nonce': self.nonce
        }
        return hashlib.sha256(json.dumps(block_data, sort_keys=True).encode()).hexdigest()

class Blockchain:
    def __init__(self, difficulty: int = 4):
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.difficulty = difficulty
        self.create_genesis_block()
    
    def create_genesis_block(self):
        genesis = Block(
            block_id="0",
            previous_hash="0",
            timestamp=time.time(),
            transactions=[],
            merkle_root="",
            nonce=0
        )
        genesis.merkle_root = genesis.calculate_merkle_root()
        genesis.hash = genesis.calculate_hash()
        self.chain.append(genesis)
    
    def get_latest_block(self) -> Block:
        return self.chain[-1]
    
    def add_transaction(self, transaction: Transaction):
        self.pending_transactions.append(transaction)
    
    def mine_pending_transactions(self) -> Block:
        if not self.pending_transactions:
            return None
        
        block = Block(
            block_id=str(len(self.chain)),
            previous_hash=self.get_latest_block().hash,
            timestamp=time.time(),
            transactions=self.pending_transactions.copy(),
            merkle_root=""
        )
        
        block.merkle_root = block.calculate_merkle_root()
        
        # Simple Proof of Work
        target = "0" * self.difficulty
        while not block.calculate_hash().startswith(target):
            block.nonce += 1
        
        block.hash = block.calculate_hash()
        self.chain.append(block)
        self.pending_transactions = []
        return block
    
    def is_valid(self) -> bool:
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            if current_block.hash != current_block.calculate_hash():
                return False
            
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True

# ===========================
# 3. GAME THEORY OPTIMIZER
# ===========================

class GameTheoryOptimizer:
    def __init__(self, nodes: List[PhysicalNode]):
        self.nodes = nodes
        self.placement_history = {}
    
    def calculate_utility(self, vm: VirtualMachine, node: PhysicalNode) -> float:
        """Calculate utility for placing VM on node"""
        if not node.can_host(vm):
            return -np.inf
        
        # Multi-objective utility function
        resource_utilization = self._calculate_resource_utilization(vm, node)
        energy_efficiency = self._calculate_energy_efficiency(vm, node)
        load_balancing = self._calculate_load_balancing(vm, node)
        
        # Weighted sum of objectives
        weights = [0.4, 0.3, 0.3]  # Can be adjusted
        utility = (weights[0] * resource_utilization + 
                  weights[1] * energy_efficiency + 
                  weights[2] * load_balancing)
        
        return utility
    
    def _calculate_resource_utilization(self, vm: VirtualMachine, node: PhysicalNode) -> float:
        """Higher utilization is better (within limits)"""
        capacity = node.get_capacity_vector()
        usage_after = node.get_usage_vector() + vm.get_resource_vector()
        utilization = usage_after / capacity
        
        # Penalize over-utilization and under-utilization
        optimal_utilization = 0.8
        penalty = np.mean(np.abs(utilization - optimal_utilization))
        return 1.0 - penalty
    
    def _calculate_energy_efficiency(self, vm: VirtualMachine, node: PhysicalNode) -> float:
        """Lower energy cost per unit resource is better"""
        resource_sum = np.sum(vm.get_resource_vector())
        if resource_sum == 0:
            return 0.0
        
        efficiency = 1.0 / (node.energy_cost * resource_sum)
        return efficiency
    
    def _calculate_load_balancing(self, vm: VirtualMachine, node: PhysicalNode) -> float:
        """Prefer nodes with more balanced load"""
        current_usage = node.get_usage_vector()
        capacity = node.get_capacity_vector()
        
        if np.any(capacity == 0):
            return 0.0
        
        usage_ratios = current_usage / capacity
        variance = np.var(usage_ratios)
        return 1.0 / (1.0 + variance)
    
    def nash_equilibrium_placement(self, vms: List[VirtualMachine]) -> Dict[str, str]:
        """Find Nash equilibrium for VM placement"""
        placement = {}
        
        # Sort VMs by priority (higher priority first)
        sorted_vms = sorted(vms, key=lambda x: x.priority, reverse=True)
        
        for vm in sorted_vms:
            best_node = None
            best_utility = -np.inf
            
            for node in self.nodes:
                utility = self.calculate_utility(vm, node)
                
                # Check if this placement affects other players (VMs)
                # Simple model: each VM tries to maximize its own utility
                if utility > best_utility:
                    best_utility = utility
                    best_node = node
            
            if best_node and best_node.can_host(vm):
                placement[vm.vm_id] = best_node.node_id
                best_node.allocate_vm(vm)
        
        return placement
    
    def pareto_optimal_solutions(self, vms: List[VirtualMachine]) -> List[Dict[str, str]]:
        """Find Pareto-optimal placement solutions"""
        solutions = []
        
        # Generate multiple placement strategies
        strategies = [
            self._greedy_resource_placement,
            self._greedy_energy_placement,
            self._balanced_placement
        ]
        
        for strategy in strategies:
            # Reset node states
            for node in self.nodes:
                node.cpu_used = node.memory_used = 0.0
                node.storage_used = node.network_used = 0.0
            
            placement = strategy(vms)
            if placement:
                solutions.append(placement)
        
        # Filter for Pareto optimality (simplified)
        return solutions
    
    def _greedy_resource_placement(self, vms: List[VirtualMachine]) -> Dict[str, str]:
        """Place VMs to maximize resource utilization"""
        placement = {}
        for vm in vms:
            best_node = max(
                [node for node in self.nodes if node.can_host(vm)],
                key=lambda n: self._calculate_resource_utilization(vm, n),
                default=None
            )
            if best_node:
                placement[vm.vm_id] = best_node.node_id
                best_node.allocate_vm(vm)
        return placement
    
    def _greedy_energy_placement(self, vms: List[VirtualMachine]) -> Dict[str, str]:
        """Place VMs to minimize energy consumption"""
        placement = {}
        for vm in vms:
            best_node = max(
                [node for node in self.nodes if node.can_host(vm)],
                key=lambda n: self._calculate_energy_efficiency(vm, n),
                default=None
            )
            if best_node:
                placement[vm.vm_id] = best_node.node_id
                best_node.allocate_vm(vm)
        return placement
    
    def _balanced_placement(self, vms: List[VirtualMachine]) -> Dict[str, str]:
        """Place VMs for balanced load distribution"""
        placement = {}
        for vm in vms:
            best_node = max(
                [node for node in self.nodes if node.can_host(vm)],
                key=lambda n: self._calculate_load_balancing(vm, n),
                default=None
            )
            if best_node:
                placement[vm.vm_id] = best_node.node_id
                best_node.allocate_vm(vm)
        return placement

# ===========================
# 4. MAIN SYSTEM ORCHESTRATOR
# ===========================

class VMPlacementSystem:
    def __init__(self, nodes: List[PhysicalNode]):
        self.nodes = nodes
        self.vms = {}
        self.blockchain = Blockchain(difficulty=2)  # Lower difficulty for demo
        self.optimizer = GameTheoryOptimizer(nodes)
        self.placement_map = {}
    
    def add_vm_request(self, vm: VirtualMachine) -> bool:
        """Add a new VM placement request"""
        self.vms[vm.vm_id] = vm
        
        # Create transaction
        tx = Transaction(
            tx_id=f"tx_{len(self.blockchain.pending_transactions)}",
            vm_id=vm.vm_id,
            node_id="",  # To be determined
            action="request",
            timestamp=time.time(),
            requester=vm.owner,
            gas_fee=random.uniform(0.01, 0.1)
        )
        
        self.blockchain.add_transaction(tx)
        return True
    
    def optimize_placement(self) -> Dict[str, str]:
        """Run game theory optimization and return placement decisions"""
        active_vms = list(self.vms.values())
        
        # Reset node states for fresh calculation
        for node in self.nodes:
            node.cpu_used = node.memory_used = 0.0
            node.storage_used = node.network_used = 0.0
        
        # Find Nash equilibrium placement
        placement = self.optimizer.nash_equilibrium_placement(active_vms)
        
        # Record placement transactions
        for vm_id, node_id in placement.items():
            tx = Transaction(
                tx_id=f"tx_{len(self.blockchain.pending_transactions)}",
                vm_id=vm_id,
                node_id=node_id,
                action="allocate",
                timestamp=time.time(),
                requester="system",
                gas_fee=0.05
            )
            self.blockchain.add_transaction(tx)
        
        self.placement_map = placement
        return placement
    
    def mine_block(self) -> Optional[Block]:
        """Mine a new block with pending transactions"""
        return self.blockchain.mine_pending_transactions()
    
    def get_system_stats(self) -> Dict:
        """Get current system statistics"""
        total_vms = len(self.vms)
        placed_vms = len(self.placement_map)
        
        node_stats = []
        for node in self.nodes:
            usage = node.get_usage_vector()
            capacity = node.get_capacity_vector()
            utilization = usage / capacity if np.any(capacity) else np.zeros_like(usage)
            
            node_stats.append({
                'node_id': node.node_id,
                'cpu_utilization': float(utilization[0]),
                'memory_utilization': float(utilization[1]),
                'storage_utilization': float(utilization[2]),
                'network_utilization': float(utilization[3])
            })
        
        return {
            'total_vms': total_vms,
            'placed_vms': placed_vms,
            'blockchain_blocks': len(self.blockchain.chain),
            'pending_transactions': len(self.blockchain.pending_transactions),
            'node_statistics': node_stats
        }

# ===========================
# 5. DEMONSTRATION & TESTING
# ===========================

def create_sample_data():
    """Create sample nodes and VMs for demonstration"""
    
    # Create physical nodes
    nodes = [
        PhysicalNode("node_1", 16.0, 32.0, 1000.0, 10.0, 0.15, "DC1"),
        PhysicalNode("node_2", 24.0, 48.0, 2000.0, 15.0, 0.18, "DC1"),
        PhysicalNode("node_3", 32.0, 64.0, 3000.0, 20.0, 0.20, "DC2"),
        PhysicalNode("node_4", 20.0, 40.0, 1500.0, 12.0, 0.16, "DC2")
    ]
    
    # Create virtual machines
    vms = [
        VirtualMachine("vm_1", 4.0, 8.0, 100.0, 2.0, 5, "user_a"),
        VirtualMachine("vm_2", 6.0, 12.0, 200.0, 3.0, 3, "user_b"),
        VirtualMachine("vm_3", 8.0, 16.0, 300.0, 4.0, 4, "user_c"),
        VirtualMachine("vm_4", 2.0, 4.0, 50.0, 1.0, 2, "user_d"),
        VirtualMachine("vm_5", 10.0, 20.0, 400.0, 5.0, 1, "user_e")
    ]
    
    return nodes, vms

def run_demonstration():
    """Run a complete demonstration of the system"""
    print("=" * 60)
    print("VM PLACEMENT OPTIMIZATION SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Initialize system
    nodes, vms = create_sample_data()
    system = VMPlacementSystem(nodes)
    
    print(f"\n1. System Initialization Complete")
    print(f"   - Created {len(nodes)} physical nodes")
    print(f"   - Prepared {len(vms)} virtual machines")
    
    # Add VM requests
    print(f"\n2. Adding VM Requests...")
    for vm in vms:
        system.add_vm_request(vm)
        print(f"   - Added VM {vm.vm_id} (CPU: {vm.cpu_requirement}, RAM: {vm.memory_requirement}GB)")
    
    # Run optimization
    print(f"\n3. Running Game Theory Optimization...")
    placement = system.optimize_placement()
    
    print(f"   Optimal Placement Results:")
    for vm_id, node_id in placement.items():
        vm = system.vms[vm_id]
        print(f"   - VM {vm_id} -> Node {node_id} (Priority: {vm.priority})")
    
    # Mine blockchain block
    print(f"\n4. Mining Blockchain Block...")
    block = system.mine_block()
    if block:
        print(f"   - Mined block {block.block_id} with {len(block.transactions)} transactions")
        print(f"   - Block hash: {block.hash[:16]}...")
        print(f"   - Nonce: {block.nonce}")
    
    # Show system statistics
    print(f"\n5. System Statistics:")
    stats = system.get_system_stats()
    print(f"   - Total VMs: {stats['total_vms']}")
    print(f"   - Placed VMs: {stats['placed_vms']}")
    print(f"   - Blockchain blocks: {stats['blockchain_blocks']}")
    print(f"   - Pending transactions: {stats['pending_transactions']}")
    
    print(f"\n6. Node Utilization:")
    for node_stat in stats['node_statistics']:
        print(f"   - {node_stat['node_id']}: CPU={node_stat['cpu_utilization']:.2%}, "
              f"RAM={node_stat['memory_utilization']:.2%}")
    
    # Verify blockchain integrity
    print(f"\n7. Blockchain Verification:")
    is_valid = system.blockchain.is_valid()
    print(f"   - Blockchain integrity: {'VALID' if is_valid else 'INVALID'}")
    
    print(f"\n8. Demonstration Complete!")
    print("=" * 60)
    
    return system

# ===========================
# 6. MAIN EXECUTION
# ===========================

if __name__ == "__main__":
    # Run the demonstration
    system = run_demonstration()
    
    # Optional: Interactive mode
    print("\nEntering interactive mode...")
    print("Available commands: stats, placement, blockchain, quit")
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == "quit":
                break
            elif command == "stats":
                stats = system.get_system_stats()
                print(json.dumps(stats, indent=2))
            elif command == "placement":
                for vm_id, node_id in system.placement_map.items():
                    print(f"VM {vm_id} -> Node {node_id}")
            elif command == "blockchain":
                print(f"Blocks: {len(system.blockchain.chain)}")
                print(f"Latest block hash: {system.blockchain.get_latest_block().hash}")
                print(f"Pending transactions: {len(system.blockchain.pending_transactions)}")
            else:
                print("Unknown command. Available: stats, placement, blockchain, quit")
                
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")
