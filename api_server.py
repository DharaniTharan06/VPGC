import hashlib
import json
from time import time
from flask import Flask, jsonify, request
from flask_cors import CORS


class PhysicalNode:
    """Represents a physical server node."""
    def __init__(self, node_id, cpu, memory, storage, network, cost, dc):
        self.node_id = node_id
        self.total_cpu = cpu
        self.total_memory = memory
        self.total_storage = storage
        self.total_network = network
        self.cost_per_hour = cost
        self.data_center = dc
        self.available_cpu = cpu
        self.available_memory = memory
        self.available_storage = storage
        self.available_network = network

class VirtualMachine:
    """Represents a virtual machine request."""
    def __init__(self, vm_id, cpu_requirement, memory_requirement, storage_requirement, network_requirement, priority, owner):
        self.vm_id = vm_id
        self.cpu = cpu_requirement
        self.memory = memory_requirement
        self.storage = storage_requirement
        self.network = network_requirement
        self.priority = priority
        self.owner = owner

class Block:
    """Represents a single block in the blockchain."""
    def __init__(self, block_id, transactions, timestamp, previous_hash):
        self.block_id = block_id
        self.transactions = transactions
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        """Calculates the hash of the block."""
        block_string = json.dumps(self.__dict__, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

class Blockchain:
    """Manages the chain of blocks."""
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.create_genesis_block()

    def create_genesis_block(self):
        """Creates the very first block in the chain."""
        genesis_block = Block(0, [], time(), "0")
        self.chain.append(genesis_block)

    def get_latest_block(self):
        """Returns the most recent block in the chain."""
        return self.chain[-1] if self.chain else None

    def add_transaction(self, transaction):
        """Adds a new transaction to the list of pending transactions."""
        self.pending_transactions.append(transaction)

    def mine(self):
        """Mines a new block, adds it to the chain, and clears pending transactions."""
        if not self.pending_transactions:
            return None 

        latest_block = self.get_latest_block()
        new_block = Block(
            block_id=latest_block.block_id + 1,
            transactions=self.pending_transactions,
            timestamp=time(),
            previous_hash=latest_block.hash
        )
        self.pending_transactions = []
        self.chain.append(new_block)
        return new_block
    
    def is_valid(self):
        """Placeholder for chain validation logic."""
        return True 

class VMPlacementSystem:
    """Manages the overall system state."""
    def __init__(self, nodes):
        self.nodes = {node.node_id: node for node in nodes}
        self.pending_vm_requests = []
        self.placed_vms = {} 
        self.blockchain = Blockchain()

    def add_vm_request(self, vm):
        """Adds a VM request and logs it as a transaction."""
        self.pending_vm_requests.append(vm)
        transaction = {'type': 'VM_REQUEST', 'vm_id': vm.vm_id, 'details': vm.__dict__}
        self.blockchain.add_transaction(transaction)
        return True

    def optimize_placement(self):
        """Simple first-fit placement algorithm that updates node resources."""
        placement_results = {}
        vms_to_remove = []

        for vm in self.pending_vm_requests:
            placed = False
            for node_id, node in self.nodes.items():
                if (node.available_cpu >= vm.cpu and 
                    node.available_memory >= vm.memory and
                    node.available_storage >= vm.storage):
                    node.available_cpu -= vm.cpu
                    node.available_memory -= vm.memory
                    node.available_storage -= vm.storage
                    self.placed_vms[vm.vm_id] = node_id
                    placement_results[vm.vm_id] = node_id
                    vms_to_remove.append(vm)
                    placed = True
                    break
            
            if not placed:
                placement_results[vm.vm_id] = None

        self.pending_vm_requests = [vm for vm in self.pending_vm_requests if vm not in vms_to_remove]
        transaction = {'type': 'PLACEMENT_RESULT', 'placement': placement_results}
        self.blockchain.add_transaction(transaction)
        return placement_results

    def mine_block(self):
        """Mines a new block with the current pending transactions."""
        return self.blockchain.mine()

    def get_system_stats(self):
        """Returns comprehensive statistics for the frontend dashboard."""
        
        node_stats = []
        for node_id, node in self.nodes.items():
            cpu_util = ((node.total_cpu - node.available_cpu) / node.total_cpu) if node.total_cpu > 0 else 0
            mem_util = ((node.total_memory - node.available_memory) / node.total_memory) if node.total_memory > 0 else 0
            
            node_stats.append({
                'node_id': node_id,
                'cpu_utilization': cpu_util,
                'memory_utilization': mem_util
            })

        return {
            'pending_vm_requests': len(self.pending_vm_requests),
            'placed_vms': len(self.placed_vms),
            'blockchain_blocks': len(self.blockchain.chain),
            'pending_transactions': len(self.blockchain.pending_transactions),
            'node_statistics': node_stats 
        }


app = Flask(__name__)
CORS(app)

nodes = [
    PhysicalNode("node_1", 16.0, 32.0, 1000.0, 10.0, 0.15, "DC1"),
    PhysicalNode("node_2", 24.0, 48.0, 2000.0, 15.0, 0.18, "DC1"),
    PhysicalNode("node_3", 32.0, 64.0, 3000.0, 20.0, 0.20, "DC2")
]
system = VMPlacementSystem(nodes)


@app.route('/api/vm/request', methods=['POST'])
def request_vm_placement():
    """Handles a new virtual machine placement request."""
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'Invalid request. JSON body required.'}), 400

    required_fields = ['vm_id', 'cpu', 'memory', 'storage', 'network', 'priority', 'owner']
    for field in required_fields:
        if field not in data:
            return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400

    try:
        vm = VirtualMachine(
            vm_id=str(data['vm_id']),
            cpu_requirement=float(data['cpu']),
            memory_requirement=float(data['memory']),
            storage_requirement=float(data['storage']),
            network_requirement=float(data['network']),
            priority=int(data['priority']),
            owner=str(data['owner'])
        )
    except (ValueError, TypeError) as e:
        return jsonify({'success': False, 'error': f'Invalid data type for a field. Details: {e}'}), 400
    
    result = system.add_vm_request(vm)
    return jsonify({'success': result, 'vm_id': vm.vm_id}), 201


@app.route('/api/optimize', methods=['POST'])
def optimize_placement():
    """Triggers the VM placement optimization and mines a new block."""
    placement = system.optimize_placement()
    block = system.mine_block()
    
    return jsonify({
        'placement': placement,
        'block_id': block.block_id if block else None,
        'transactions_in_block': len(block.transactions) if block else 0
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Returns current statistics of the system."""
    stats = system.get_system_stats()
    return jsonify(stats)


@app.route('/api/blockchain', methods=['GET'])
def get_blockchain_info():
    """Returns information about the blockchain."""
    latest_block = system.blockchain.get_latest_block()
    latest_block_hash = latest_block.hash if latest_block else None
    
    return jsonify({
        'blocks_in_chain': len(system.blockchain.chain),
        'pending_transactions': len(system.blockchain.pending_transactions),
        'is_chain_valid': system.blockchain.is_valid(),
        'latest_block_hash': latest_block_hash
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)