"""Create remaining Hydra config YAML files."""

import os

CONFIGS = {
    "configs/experiment_tracking/mlflow.yaml": (
        'backend: "mlflow"\n'
        'tracking_uri: ""\n'
        'project_name: "theo"\n'
    ),
    "configs/experiment_tracking/wandb.yaml": (
        'backend: "wandb"\n'
        'project_name: "theo"\n'
    ),
    "configs/memory/default.yaml": (
        "working_memory_capacity: 100\n"
        "consolidation_interval_seconds: 3600\n"
        "forgetting_enabled: true\n"
        'default_backend: "in_memory"\n'
    ),
    "configs/knowledge/default.yaml": (
        'graph_backend: "in_memory"\n'
        "max_traversal_depth: 5\n"
    ),
    "configs/identity/default.yaml": (
        'persona_name: "Theo"\n'
        "consistency_check_enabled: true\n"
    ),
    "configs/goals/default.yaml": (
        "max_active_goals: 10\n"
        'default_priority: "medium"\n'
    ),
    "configs/scheduler/default.yaml": "enabled: true\n" 'timezone: "UTC"\n',
    "configs/security/default.yaml": "sandbox_enabled: false\n" 'secret_backend: "env"\n',
    "configs/telemetry/default.yaml": (
        "enabled: true\n"
        "metrics_enabled: true\n"
        "tracing_enabled: false\n"
        "health_check_interval_seconds: 60\n"
    ),
    "configs/evaluation/default.yaml": "default_benchmarks: []\n" 'reports_dir: "reports"\n',
    "configs/dataset/default.yaml": 'data_dir: "data"\n' 'default_format: "json"\n',
    "configs/kernel/default.yaml": "boot_timeout_seconds: 30\n",
    "configs/perception/default.yaml": 'default_modality: "text"\n',
    "configs/context/default.yaml": "max_context_items: 50\n",
}

for path, content in CONFIGS.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

print(f"Created {len(CONFIGS)} config files")
