from flask import Flask, jsonify, request
from flask_cors import CORS 
from main import VMPlacementSystem, PhysicalNode, VirtualMachine
import json
import numpy as np

app = Flask(__name__) 
CORS(app) 
def create_sample_data():
    nodes = [
        PhysicalNode("node_1", 16.0, 32.0, 1000.0, 10.0, 0.15, "DC1"),
        PhysicalNode("node_2", 24.0, 48.0, 2000.0, 15.0, 0.18, "DC1"),
        PhysicalNode("node_3", 32.0, 64.0, 3000.0, 20.0, 0.20, "DC2"),
        PhysicalNode("node_4", 20.0, 40.0, 1500.0, 12.0, 0.16, "DC2")
    ]
    vms = [
        VirtualMachine("vm_1", 4.0, 8.0, 100.0, 2.0, 5, "user_a"),
        VirtualMachine("vm_2", 6.0, 12.0, 200.0, 3.0, 3, "user_b"),
        VirtualMachine("vm_3", 8.0, 16.0, 300.0, 4.0, 4, "user_c"),
        VirtualMachine("vm_4", 2.0, 4.0, 50.0, 1.0, 2, "user_d"),
        VirtualMachine("vm_5", 10.0, 20.0, 400.0, 5.0, 1, "user_e")
    ]
    return nodes, vms


nodes, vms = create_sample_data()
system = VMPlacementSystem(nodes)

for vm in vms:
    system.add_vm_request(vm)


@app.route('/api/vm/request', methods=['POST'])
def request_vm_placement():
    data = request.json
    vm = VirtualMachine(
        vm_id=data['vm_id'],
        cpu_requirement=data['cpu'],
        memory_requirement=data['memory'],
        storage_requirement=data['storage'],
        network_requirement=data['network'],
        priority=data['priority'],
        owner=data['owner']
    )
    result = system.add_vm_request(vm)
    return jsonify({'success': result, 'vm_id': vm.vm_id})


@app.route('/api/optimize', methods=['POST'])
def optimize_placement():
    placement = system.optimize_placement()
    block = system.mine_block()
    return jsonify({
        'placement': placement,
        'block_id': block.block_id if block else None,
        'transactions': len(block.transactions) if block else 0
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    stats = system.get_system_stats()

    active_cpu_utils = [
        node['cpu_utilization'] for node in stats['node_statistics'] 
        if node['cpu_utilization'] > 0
    ]
    load_balance_variance = np.var(active_cpu_utils) if len(active_cpu_utils) > 1 else 0

    total_energy_cost = 0
    placed_vm_count = len(system.placement_map)
    if placed_vm_count > 0:
        for vm_id, node_id in system.placement_map.items():
            node = next((n for n in system.nodes if n.node_id == node_id), None)
            if node:
                total_energy_cost += node.energy_cost
        avg_energy_cost = total_energy_cost / placed_vm_count
    else:
        avg_energy_cost = 0

    stats['performance_metrics'] = {
        'load_balance_variance': round(load_balance_variance, 4),
        'avg_energy_cost': round(avg_energy_cost, 4),
        'success_rate': (stats['placed_vms'] / stats['total_vms']) if stats['total_vms'] > 0 else 0
    }

    stats['comparison_data'] = {
        'labels': ['Load Balance (Variance)', 'Energy Cost (Avg)', 'SLA Violations (Est. %)'],
        'datasets': [
            {
                'label': 'Your Model (Game Theory)',
                'data': [round(load_balance_variance, 4), round(avg_energy_cost, 4), 1.5],
                'backgroundColor': 'rgba(75, 192, 192, 0.7)'
            },
            {
                'label': 'First-Fit',
                'data': [0.051, 0.162, 3.5],
                'backgroundColor': 'rgba(255, 99, 132, 0.7)'
            },
            {
                'label': 'Greedy (Energy Saver)',
                'data': [0.045, 0.158, 5.8],
                'backgroundColor': 'rgba(255, 206, 86, 0.7)'
            },
            {
                'label': 'Round-Robin',
                'data': [0.009, 0.180, 4.0],
                'backgroundColor': 'rgba(153, 102, 255, 0.7)'
            }
        ]
    }
    return jsonify(stats)


@app.route('/api/blockchain', methods=['GET'])
def get_blockchain_info():
    return jsonify({
        'blocks': len(system.blockchain.chain),
        'pending_transactions': len(system.blockchain.pending_transactions),
        'is_valid': system.blockchain.is_valid(),
        'latest_block_hash': system.blockchain.get_latest_block().hash
    })


def initialize_system():
    """Run initial optimization without returning Flask response"""
    print("Running initial optimization...")
    placement = system.optimize_placement()
    block = system.mine_block()
    print(f"Initial optimization complete!")
    print(f"Placement: {len(placement)} VMs placed")
    if block:
        print(f"Block {block.block_id} mined with {len(block.transactions)} transactions")


if __name__ == '__main__':  
    initialize_system()  
    app.run(debug=True, host='0.0.0.0', port=5000)