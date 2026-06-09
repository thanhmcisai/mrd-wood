#!/usr/bin/env python3
"""
Full experiment runner for Colab/L4.

Example:
    python scripts/run_full_colab.py --config configs/full_colab_l4.json
    python scripts/run_full_colab.py --config configs/full_colab_l4.json --resume
    python scripts/run_full_colab.py --config configs/full_colab_l4.json --only exp4 exp9
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Set


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EXPERIMENT_MODULES = {
    'exp1': 'wood_spatial.experiments.exp1_robustness',
    'exp1b': 'wood_spatial.experiments.exp1b_feature_geometry',
    'exp2': 'wood_spatial.experiments.exp2_spatial_clustering',
    'exp3': 'wood_spatial.experiments.exp3_gradcam_cluster',
    'exp4': 'wood_spatial.experiments.exp4_cross_dataset',
    'exp5': 'wood_spatial.experiments.exp5_magnification',
    'exp5_full_crossmag': 'wood_spatial.experiments.exp5_full_crossmag',
    'exp5b': 'wood_spatial.experiments.exp5b_crossmag_spatial',
    'exp6': 'wood_spatial.experiments.exp6_multilevel_failure',
    'exp7': 'wood_spatial.experiments.exp7_path_analysis',
    'exp7_lodo': 'wood_spatial.experiments.exp7_lodo_feature_drift',
    'exp8': 'wood_spatial.experiments.exp8_failure_detection',
    'exp8_baselines': 'wood_spatial.experiments.exp8_baselines',
    'exp9': 'wood_spatial.experiments.exp9_tierb_validation',
    'exp10': 'wood_spatial.experiments.exp10_reference_monitor',
    'exp10_lopo': 'wood_spatial.experiments.exp10_lopo_monitor',
    'exp10_lodo': 'wood_spatial.experiments.exp10_lodo_monitor',
    'exp10_sensitivity': 'wood_spatial.experiments.exp10_monitor_sensitivity',
    'exp10_operating_points': 'wood_spatial.experiments.exp10_operating_points',
    'exp_k_ablation': 'wood_spatial.experiments.exp_k_ablation',
    'exp_mixed_effects': 'wood_spatial.experiments.exp_mixed_effects',
    'exp_knn_sensitivity': 'wood_spatial.experiments.exp_knn_sensitivity',
    'exp_competitor_switching': 'wood_spatial.experiments.exp_competitor_switching',
    'exp_augmentation_probe': 'wood_spatial.experiments.exp_augmentation_probe',
    'exp7_controlled_hierarchical': 'wood_spatial.experiments.exp7_controlled_hierarchical',
    'exp_ci_main_tables': 'wood_spatial.experiments.exp_ci_main_tables',
    'run_ablations': 'wood_spatial.experiments.run_ablations',
    'hires_spatial': 'wood_spatial.experiments.exp_hires_spatial_full',
    'hires_extract': 'wood_spatial.experiments.exp_hires_extract',
    'hires_metrics': 'wood_spatial.experiments.exp_hires_metrics_from_cache',
    'fig_vn26': 'wood_spatial.figures.regen_figs',
    'fig_vn26_perturbation': 'wood_spatial.figures.regen_fig4b',
}

POST_EXTRACT_WAVES = [
    ['exp1'],
    ['exp1b', 'exp2', 'exp3', 'exp5', 'exp5b', 'exp8_baselines'],
    ['exp4', 'exp9'],
    ['exp6'],
    ['exp7', 'exp8'],
    ['exp10'],
    ['exp5_full_crossmag', 'exp7_lodo', 'exp10_lopo', 'exp10_lodo', 'exp_mixed_effects'],
    ['exp10_sensitivity'],
    ['exp_knn_sensitivity', 'exp10_operating_points', 'exp7_controlled_hierarchical', 'exp_ci_main_tables'],
    ['exp_k_ablation'],
    ['run_ablations'],
    ['hires_spatial'],
    ['exp_competitor_switching', 'exp_augmentation_probe'],
    ['fig_vn26', 'fig_vn26_perturbation'],
]


def load_config(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def configure_environment(cfg: dict):
    paths = cfg['paths']
    os.environ['WOOD_BASE'] = paths['root_path']
    os.environ['WOOD_DATASETS_DIR'] = paths['datasets_dir']
    os.environ['WOOD_RESULTS_DIR'] = paths['results_dir']
    os.environ['WOOD_NUM_WORKERS'] = str(cfg.get('runtime', {}).get('num_workers', 0))
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ.setdefault(
        'PYTORCH_CUDA_ALLOC_CONF',
        cfg.get('runtime', {}).get('torch_cuda_alloc_conf', 'expandable_segments:True'),
    )
    hi = cfg.get('highres_spatial', {})
    if hi:
        if hi.get('datasets'):
            os.environ['WOOD_HIRES_DATASETS'] = ','.join(hi['datasets'])
        if hi.get('backbones'):
            os.environ['WOOD_HIRES_BACKBONES'] = ','.join(hi['backbones'])
        if hi.get('conditions'):
            os.environ['WOOD_HIRES_CONDITIONS'] = str(hi['conditions'])
        if hi.get('n_per_class') is not None:
            os.environ['WOOD_HIRES_N_PER_CLASS'] = str(hi['n_per_class'])
        if hi.get('img_w') is not None:
            os.environ['WOOD_HIRES_IMG_W'] = str(hi['img_w'])
        if hi.get('img_h') is not None:
            os.environ['WOOD_HIRES_IMG_H'] = str(hi['img_h'])
        if hi.get('n_fit') is not None:
            os.environ['WOOD_HIRES_N_FIT'] = str(hi['n_fit'])
        if hi.get('n_init') is not None:
            os.environ['WOOD_HIRES_N_INIT'] = str(hi['n_init'])

    for key in ('results_dir',):
        Path(paths[key]).mkdir(parents=True, exist_ok=True)


def state_path(cfg: dict) -> Path:
    return Path(cfg['paths']['results_dir']) / 'full_run_state.json'


def load_state(cfg: dict) -> dict:
    p = state_path(cfg)
    if not p.exists():
        return {'completed': []}
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_state(cfg: dict, state: dict):
    p = state_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


def mark_done(cfg: dict, state: dict, stage: str):
    if stage not in state['completed']:
        state['completed'].append(stage)
    state['last_completed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    save_state(cfg, state)


def should_skip(stage: str, state: dict, resume: bool, selected: Optional[Set[str]]) -> bool:
    if selected is not None and stage not in selected:
        return True
    return resume and stage in state.get('completed', [])


def run_module(module: str, extra_args: Optional[list] = None):
    cmd = [sys.executable, '-u', '-m', module]
    if extra_args:
        cmd.extend(extra_args)
    print('\n>>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _log_path(cfg: dict, stage: str) -> Path:
    log_dir = Path(cfg['paths']['results_dir']) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f'{stage}.log'


def _print_log_tail(log_file: Path, n_lines: int = 80):
    if not log_file.exists():
        print(f'  log missing: {log_file}', flush=True)
        return
    try:
        lines = log_file.read_text(errors='replace').splitlines()
    except Exception as exc:
        print(f'  could not read log {log_file}: {exc}', flush=True)
        return
    print(f'\n--- tail {min(n_lines, len(lines))} lines: {log_file} ---', flush=True)
    for line in lines[-n_lines:]:
        print(line, flush=True)
    print('--- end log tail ---\n', flush=True)


def _safe_close(handle):
    try:
        handle.close()
    except OSError as exc:
        print(f'  WARN could not close log handle cleanly: {exc}', flush=True)


def run_parallel_stages(cfg: dict, stages: list, jobs: int, state: dict, resume: bool):
    pending = [
        stage for stage in stages
        if not (resume and stage in state.get('completed', []))
    ]
    if not pending:
        print('All stages in this wave are already complete.', flush=True)
        return

    running = []
    failures = []
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    env.setdefault('OPENBLAS_NUM_THREADS', '1')

    while pending or running:
        while pending and len(running) < jobs:
            stage = pending.pop(0)
            module = EXPERIMENT_MODULES[stage]
            cmd = [sys.executable, '-u', '-m', module]
            if stage == 'exp10':
                exp10_jobs = int(cfg.get('runtime', {}).get('exp10_jobs', jobs))
                cmd.extend(['--jobs', str(max(1, exp10_jobs))])
            if stage == 'exp10_sensitivity':
                sens = cfg.get('monitor_sensitivity', {})
                sens_jobs = int(sens.get('jobs', cfg.get('runtime', {}).get('exp10_sensitivity_jobs', jobs)))
                cmd.extend(['--jobs', str(max(1, sens_jobs))])
                if sens.get('batch_sizes'):
                    cmd.extend(['--batch-sizes', ','.join(str(x) for x in sens['batch_sizes'])])
                if sens.get('ref_sizes'):
                    cmd.extend(['--ref-sizes', ','.join(str(x) for x in sens['ref_sizes'])])
                if sens.get('repeats') is not None:
                    cmd.extend(['--repeats', str(sens['repeats'])])
            if stage == 'exp_knn_sensitivity':
                knn_jobs = int(cfg.get('runtime', {}).get('exp_knn_sensitivity_jobs', jobs))
                cmd.extend(['--jobs', str(max(1, knn_jobs))])
            if stage == 'exp_competitor_switching':
                comp_jobs = int(cfg.get('runtime', {}).get('exp_competitor_switching_jobs', jobs))
                cmd.extend(['--jobs', str(max(1, comp_jobs))])
            log_file = _log_path(cfg, stage)
            fh = log_file.open('w', buffering=1)
            print(f'  START {stage}: {" ".join(cmd)} | log={log_file}', flush=True)
            proc = subprocess.Popen(
                cmd,
                stdout=fh,
                stderr=subprocess.STDOUT,
                env=env,
            )
            running.append({
                'stage': stage, 'proc': proc, 'fh': fh, 'log': log_file,
                'started': time.time(), 'last_heartbeat': time.time(),
            })

        time.sleep(5)
        still_running = []
        for item in running:
            code = item['proc'].poll()
            if code is None:
                now = time.time()
                if now - item['last_heartbeat'] >= 60:
                    elapsed = (now - item['started']) / 60
                    size = item['log'].stat().st_size if item['log'].exists() else 0
                    print(f'  RUNNING {item["stage"]}: {elapsed:.1f} min | log_size={size} bytes', flush=True)
                    item['last_heartbeat'] = now
                still_running.append(item)
                continue
            _safe_close(item['fh'])
            if code == 0:
                print(f'  DONE  {item["stage"]}', flush=True)
                mark_done(cfg, state, item['stage'])
            else:
                print(f'  FAIL  {item["stage"]} code={code} log={item["log"]}', flush=True)
                _print_log_tail(item['log'])
                failures.append(item)
        running = still_running

        if failures:
            for item in running:
                item['proc'].terminate()
                _safe_close(item['fh'])
            failed = ', '.join(item['stage'] for item in failures)
            raise subprocess.CalledProcessError(1, f'parallel wave failed: {failed}')


def _deep_tasks_for_stage(cfg: dict, stage: str) -> list:
    if stage == 'extract':
        ext = cfg['extraction']
        backbones = cfg['backbones']
        clean_datasets = list(ext['global_clean_datasets'])
        perturb_datasets = set(ext['global_perturbation_datasets'])
        spatial_clean = list(ext.get('spatial_clean_datasets', []))
        spatial_perturb_datasets = set(ext.get('spatial_perturbation_datasets', []))
        spatial_specs = [
            f'{name}:{value}'
            for name, value in ext.get('spatial_perturbations', [])
        ]

        tasks = []
        for ds in clean_datasets:
            for bb in backbones:
                cmd = [
                    sys.executable, '-u', '-m', 'wood_spatial.experiments.extract_features',
                    '--mode', 'global',
                    '--clean-datasets', ds,
                    '--perturb-datasets', ds if ds in perturb_datasets else 'none',
                    '--backbones', bb,
                ]
                tasks.append({'name': f'extract_global_{ds}_{bb}', 'cmd': cmd})

        for ds in spatial_clean:
            specs = spatial_specs if ds in spatial_perturb_datasets else ['none']
            for bb in backbones:
                cmd = [
                    sys.executable, '-u', '-m', 'wood_spatial.experiments.extract_features',
                    '--mode', 'spatial',
                    '--datasets', ds,
                    '--backbones', bb,
                    '--spatial-perturbations', *specs,
                ]
                tasks.append({'name': f'extract_spatial_{ds}_{bb}', 'cmd': cmd})
        return tasks

    if stage == 'exp1':
        return [
            {
                'name': f'exp1_{ds}_{bb}',
                'cmd': [
                    sys.executable, '-u', '-m', EXPERIMENT_MODULES['exp1'],
                    '--datasets', ds, '--backbones', bb, '--checkpoint-only',
                ],
            }
            for ds in cfg['datasets']['tier_a']
            for bb in cfg['backbones']
        ]
    if stage == 'exp2':
        return [
            {
                'name': f'exp2_{ds}',
                'cmd': [
                    sys.executable, '-u', '-m', EXPERIMENT_MODULES['exp2'],
                    '--datasets', ds, '--checkpoint-only',
                ],
            }
            for ds in cfg['datasets']['tier_a']
        ]
    if stage == 'exp3':
        return [
            {
                'name': f'exp3_{ds}_{bb}',
                'cmd': [
                    sys.executable, '-u', '-m', EXPERIMENT_MODULES['exp3'],
                    '--datasets', ds, '--backbones', bb, '--checkpoint-only',
                ],
            }
            for ds in cfg['datasets']['tier_a']
            for bb in cfg['backbones']
        ]
    if stage == 'exp1b':
        return [
            {
                'name': f'exp1b_{ds}_{bb}',
                'cmd': [
                    sys.executable, '-u', '-m', EXPERIMENT_MODULES['exp1b'],
                    '--datasets', ds, '--backbones', bb, '--checkpoint-only',
                ],
            }
            for ds in cfg['datasets']['tier_a']
            for bb in cfg['backbones']
        ]
    if stage == 'exp9':
        return [
            {
                'name': f'exp9_{ds}_{bb}',
                'cmd': [
                    sys.executable, '-u', '-m', EXPERIMENT_MODULES['exp9'],
                    '--datasets', ds, '--backbones', bb, '--checkpoint-only',
                ],
            }
            for ds in cfg['datasets']['tier_b']
            for bb in cfg['backbones']
        ]
    if stage == 'exp8_baselines':
        return [
            {
                'name': f'exp8_baselines_{ds}_{bb}',
                'cmd': [
                    sys.executable, '-u', '-m', EXPERIMENT_MODULES['exp8_baselines'],
                    '--datasets', ds, '--backbones', bb, '--checkpoint-only',
                ],
            }
            for ds in cfg['datasets']['tier_a']
            for bb in cfg['backbones']
        ]
    if stage == 'exp_knn_sensitivity':
        return [
            {
                'name': f'exp_knn_sensitivity_{ds}_{bb}',
                'cmd': [
                    sys.executable, '-u', '-m', EXPERIMENT_MODULES['exp_knn_sensitivity'],
                    '--datasets', ds, '--backbones', bb,
                ],
            }
            for ds in cfg['datasets']['tier_a']
            for bb in cfg['backbones']
        ]
    if stage == 'run_ablations':
        return [
            {
                'name': f'run_ablations_deblur_{ds}_{bb}',
                'cmd': [
                    sys.executable, '-u', '-m', EXPERIMENT_MODULES['run_ablations'],
                    '--only', 'deblur',
                    '--datasets', ds,
                    '--backbones', bb,
                    '--checkpoint-only',
                ],
            }
            for ds in cfg['datasets']['tier_a']
            for bb in cfg['backbones']
        ]
    if stage == 'hires_extract':
        hi = cfg.get('highres_spatial', {})
        ds_groups = cfg.get('datasets', {})
        requested = hi.get('datasets') or ['tier_a']
        datasets = []
        for item in requested:
            if item == 'all':
                datasets.extend(ds_groups.get('tier_a', []))
                datasets.extend(ds_groups.get('tier_b', []))
                datasets.extend(ds_groups.get('tier_c', []))
            elif item == 'vn26':
                datasets.extend(ds_groups.get('tier_c', []))
            elif item in ds_groups:
                datasets.extend(ds_groups[item])
            else:
                datasets.append(item)
        seen = []
        datasets = [d for d in datasets if not (d in seen or seen.append(d))]

        backbones = [str(x) for x in hi.get('backbones', ['all'])]
        conditions = str(hi.get('conditions', 'representative'))
        n_per_class = str(hi.get('n_per_class', 3))
        compressed = bool(hi.get('compressed', False))

        tasks = []
        for ds in datasets:
            cmd = [
                sys.executable, '-u', '-m', EXPERIMENT_MODULES['hires_extract'],
                '--datasets', ds,
                '--backbones', *backbones,
                '--conditions', conditions,
                '--n-per-class', n_per_class,
            ]
            if compressed:
                cmd.append('--compressed')
            tasks.append({'name': f'hires_extract_{ds}', 'cmd': cmd})
        return tasks
    if stage == 'hires_spatial':
        hi = cfg.get('highres_spatial', {})
        ds_groups = cfg.get('datasets', {})
        requested = hi.get('datasets') or ['tier_a']
        datasets = []
        for item in requested:
            if item == 'all':
                datasets.extend(ds_groups.get('tier_a', []))
                datasets.extend(ds_groups.get('tier_b', []))
                datasets.extend(ds_groups.get('tier_c', []))
            elif item == 'vn26':
                datasets.extend(ds_groups.get('tier_c', []))
            elif item in ds_groups:
                datasets.extend(ds_groups[item])
            else:
                datasets.append(item)
        seen = []
        datasets = [d for d in datasets if not (d in seen or seen.append(d))]

        requested_bbs = hi.get('backbones') or ['all']
        if requested_bbs == ['all']:
            backbones = cfg['backbones']
        else:
            backbones = requested_bbs
        conditions = str(hi.get('conditions', 'representative'))
        n_per_class = str(hi.get('n_per_class', 3))

        return [
            {
                'name': f'hires_spatial_{ds}_{bb}',
                'cmd': [
                    sys.executable, '-u', '-m', EXPERIMENT_MODULES['hires_spatial'],
                    '--datasets', ds,
                    '--backbones', bb,
                    '--conditions', conditions,
                    '--n-per-class', n_per_class,
                ],
            }
            for ds in datasets
            for bb in backbones
        ]
    if stage == 'hires_metrics':
        hi = cfg.get('highres_spatial', {})
        ds_groups = cfg.get('datasets', {})
        requested = hi.get('datasets') or ['tier_a']
        datasets = []
        for item in requested:
            if item == 'all':
                datasets.extend(ds_groups.get('tier_a', []))
                datasets.extend(ds_groups.get('tier_b', []))
                datasets.extend(ds_groups.get('tier_c', []))
            elif item == 'vn26':
                datasets.extend(ds_groups.get('tier_c', []))
            elif item in ds_groups:
                datasets.extend(ds_groups[item])
            else:
                datasets.append(item)
        seen = []
        datasets = [d for d in datasets if not (d in seen or seen.append(d))]

        requested_bbs = hi.get('backbones') or ['all']
        if requested_bbs == ['all']:
            backbones = cfg['backbones']
        else:
            backbones = requested_bbs
        conditions = str(hi.get('conditions', 'representative'))
        n_per_class = str(hi.get('n_per_class', 3))

        return [
            {
                'name': f'hires_metrics_{ds}_{bb}',
                'cmd': [
                    sys.executable, '-u', '-m', EXPERIMENT_MODULES['hires_metrics'],
                    '--datasets', ds,
                    '--backbones', bb,
                    '--conditions', conditions,
                    '--n-per-class', n_per_class,
                ],
            }
            for ds in datasets
            for bb in backbones
        ]
    return []


def _finalize_deep_stage(stage: str):
    if stage == 'extract':
        return
    if stage == 'hires_extract':
        return
    if stage == 'hires_spatial':
        module = EXPERIMENT_MODULES[stage]
        cmd = [sys.executable, '-u', '-m', module, '--finalize-only']
        print(f'  FINALIZE {stage}: {" ".join(cmd)}', flush=True)
        subprocess.run(cmd, check=True)
        return
    if stage == 'hires_metrics':
        module = EXPERIMENT_MODULES[stage]
        cmd = [sys.executable, '-u', '-m', module, '--finalize-only']
        print(f'  FINALIZE {stage}: {" ".join(cmd)}', flush=True)
        subprocess.run(cmd, check=True)
        return
    if stage in {'exp1', 'exp1b', 'exp8_baselines', 'exp9'}:
        module = EXPERIMENT_MODULES[stage]
        cmd = [sys.executable, '-u', '-m', module, '--finalize-only']
        print(f'  FINALIZE {stage}: {" ".join(cmd)}', flush=True)
        subprocess.run(cmd, check=True)
        return
    if stage == 'exp_knn_sensitivity':
        module = EXPERIMENT_MODULES[stage]
        cmd = [sys.executable, '-u', '-m', module, '--finalize-only']
        print(f'  FINALIZE {stage}: {" ".join(cmd)}', flush=True)
        subprocess.run(cmd, check=True)
        return
    if stage == 'run_ablations':
        module = EXPERIMENT_MODULES[stage]
        # Seed and Mahalanobis are CSV-resumable and quick relative to deblur;
        # deblur was checkpointed in deep-parallel workers above.
        for cmd in [
            [sys.executable, '-u', '-m', module, '--only', 'seed', 'mahalanobis'],
            [sys.executable, '-u', '-m', module, '--only', 'deblur', '--finalize-only'],
        ]:
            print(f'  FINALIZE {stage}: {" ".join(cmd)}', flush=True)
            subprocess.run(cmd, check=True)
        return
    module = EXPERIMENT_MODULES[stage]
    cmd = [sys.executable, '-u', '-m', module]
    print(f'  FINALIZE {stage}: {" ".join(cmd)}', flush=True)
    subprocess.run(cmd, check=True)


def run_deep_stage(cfg: dict, stage: str, jobs: int, state: dict, resume: bool):
    if resume and stage in state.get('completed', []):
        print(f'SKIP {stage}', flush=True)
        return

    tasks = _deep_tasks_for_stage(cfg, stage)
    if not tasks:
        run_parallel_stages(cfg, [stage], 1, state, resume)
        return

    running = []
    failures = []
    pending = list(tasks)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    env.setdefault('OPENBLAS_NUM_THREADS', '1')

    while pending or running:
        while pending and len(running) < jobs:
            task = pending.pop(0)
            log_file = _log_path(cfg, task['name'])
            fh = log_file.open('w', buffering=1)
            print(f'  START {task["name"]}: {" ".join(task["cmd"])} | log={log_file}', flush=True)
            proc = subprocess.Popen(task['cmd'], stdout=fh, stderr=subprocess.STDOUT, env=env)
            running.append({
                'task': task, 'proc': proc, 'fh': fh, 'log': log_file,
                'started': time.time(), 'last_heartbeat': time.time(),
            })

        time.sleep(5)
        still_running = []
        for item in running:
            code = item['proc'].poll()
            if code is None:
                now = time.time()
                if now - item['last_heartbeat'] >= 60:
                    elapsed = (now - item['started']) / 60
                    size = item['log'].stat().st_size if item['log'].exists() else 0
                    print(f'  RUNNING {item["task"]["name"]}: {elapsed:.1f} min | log_size={size} bytes', flush=True)
                    item['last_heartbeat'] = now
                still_running.append(item)
                continue
            _safe_close(item['fh'])
            if code == 0:
                print(f'  DONE  {item["task"]["name"]}', flush=True)
            else:
                print(f'  FAIL  {item["task"]["name"]} code={code} log={item["log"]}', flush=True)
                _print_log_tail(item['log'])
                failures.append(item)
        running = still_running

        if failures:
            for item in running:
                item['proc'].terminate()
                _safe_close(item['fh'])
            failed = ', '.join(item['task']['name'] for item in failures)
            raise subprocess.CalledProcessError(1, f'deep stage failed: {failed}')

    _finalize_deep_stage(stage)
    mark_done(cfg, state, stage)


def run_post_extract_parallel(cfg: dict, jobs: int, state: dict, resume: bool, selected: Optional[Set[str]]):
    explicit_selection = selected is not None
    stage_resume = resume if not explicit_selection else False
    if selected is not None:
        if 'install' in selected:
            if cfg.get('runtime', {}).get('install_dependencies', False):
                install_dependencies()
            else:
                print('Dependency installation disabled in config.', flush=True)
            mark_done(cfg, state, 'install')
        if 'verify' in selected:
            verify_datasets(cfg)
            mark_done(cfg, state, 'verify')
        if 'extract' in selected:
            print('\n=== DEEP PARALLEL: extract ===', flush=True)
            run_deep_stage(cfg, 'extract', jobs=jobs, state=state, resume=stage_resume)

        selected = set(selected) - {'install', 'verify', 'extract'}
        if not selected:
            return

    waves = POST_EXTRACT_WAVES
    if selected is not None:
        waves = [[stage for stage in wave if stage in selected] for wave in waves]
        waves = [wave for wave in waves if wave]

    for idx, wave in enumerate(waves, start=1):
        print(f'\n=== PARALLEL WAVE {idx}: {" ".join(wave)} ===', flush=True)
        deep_stage_names = {
            'exp1', 'exp1b', 'exp2', 'exp3', 'exp8_baselines', 'exp9', 'exp_knn_sensitivity',
            'run_ablations', 'hires_spatial', 'hires_extract', 'hires_metrics',
        }
        deep = [stage for stage in wave if stage in deep_stage_names]
        regular = [stage for stage in wave if stage not in deep_stage_names]

        if regular:
            wave_jobs = min(jobs, len(regular))
            run_parallel_stages(cfg, regular, wave_jobs, state, stage_resume)
        for stage in deep:
            print(f'\n=== DEEP PARALLEL: {stage} ===', flush=True)
            run_deep_stage(cfg, stage, jobs=jobs, state=state, resume=stage_resume)


def verify_datasets(cfg: dict):
    from wood_spatial.config import ALL_DATASETS

    wanted = []
    ds_cfg = cfg['datasets']
    for key in ('tier_a', 'tier_b', 'tier_c'):
        wanted.extend(ds_cfg[key])

    missing = []
    for ds in wanted:
        root = Path(ALL_DATASETS[ds]['root'])
        if not root.exists():
            missing.append((ds, str(root)))

    if missing:
        print('\nMissing dataset folders:')
        for ds, root in missing:
            print(f'  {ds}: {root}')
        raise SystemExit('Dataset verification failed.')

    print('Dataset verification passed:')
    for ds in wanted:
        print(f'  {ds}: {ALL_DATASETS[ds]["root"]}')


def run_extraction(cfg: dict, force: bool):
    from wood_spatial.experiments.extract_features import (
        extract_global_features,
        extract_spatial_features,
    )

    ext = cfg['extraction']
    backbones = cfg['backbones']

    extract_global_features(
        backbones=backbones,
        clean_datasets=ext['global_clean_datasets'],
        perturbation_datasets=ext['global_perturbation_datasets'],
        force=force,
    )

    extract_spatial_features(
        datasets=ext['spatial_clean_datasets'],
        backbones=backbones,
        perturbation_subset=[],
        force=force,
    )

    spatial_pt = [tuple(x) for x in ext['spatial_perturbations']]
    extract_spatial_features(
        datasets=ext['spatial_perturbation_datasets'],
        backbones=backbones,
        perturbation_subset=spatial_pt,
        force=force,
    )


def install_dependencies():
    cmd = [
        sys.executable, '-m', 'pip', 'install', '-q',
        'timm==0.9.12',
        'opencv-python-headless',
        'scikit-learn',
        'scipy',
        'statsmodels',
        'pandas',
        'matplotlib',
        'seaborn',
        'tqdm',
    ]
    print('\n>>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description='Run full Wood Spatial experiments on Colab.')
    parser.add_argument('--config', default='configs/full_colab_l4.json')
    parser.add_argument('--resume', action='store_true',
                        help='Skip stages already marked complete in results/full_run_state.json')
    parser.add_argument('--force', action='store_true',
                        help='Force cache regeneration during extraction')
    parser.add_argument('--only', nargs='+', default=None,
                        help='Run only selected stages: install, verify, extract, or experiment ids.')
    parser.add_argument('--parallel-post', action='store_true',
                        help='Run post-extraction experiments in dependency-safe parallel waves.')
    parser.add_argument('--jobs', type=int, default=None,
                        help='Max parallel experiment processes for --parallel-post.')
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    configure_environment(cfg)

    runtime = cfg.get('runtime', {})
    resume = args.resume or bool(runtime.get('resume', False))
    force = args.force or bool(runtime.get('force', False))
    selected = set(args.only) if args.only else None
    state = load_state(cfg)
    jobs = args.jobs or int(runtime.get('parallel_jobs', 3))
    stage_resume = resume if selected is None else False

    stages = ['install', 'verify', 'extract'] + cfg['experiments']

    if args.parallel_post:
        t0 = time.time()
        run_post_extract_parallel(cfg, jobs=jobs, state=state, resume=resume, selected=selected)
        elapsed = (time.time() - t0) / 3600
        print(f'\nParallel post-extraction run finished in {elapsed:.2f} hours.')
        print(f'Results: {cfg["paths"]["results_dir"]}')
        return

    t0 = time.time()
    for stage in stages:
        if should_skip(stage, state, stage_resume, selected):
            print(f'SKIP {stage}', flush=True)
            continue

        print(f'\n=== STAGE: {stage} ===', flush=True)
        if stage == 'install':
            if runtime.get('install_dependencies', False):
                install_dependencies()
            else:
                print('Dependency installation disabled in config.')
        elif stage == 'verify':
            verify_datasets(cfg)
        elif stage == 'extract':
            run_extraction(cfg, force=force)
        else:
            module = EXPERIMENT_MODULES[stage]
            extra_args = []
            if stage == 'exp10':
                exp10_jobs = int(runtime.get('exp10_jobs', jobs))
                extra_args.extend(['--jobs', str(max(1, exp10_jobs))])
            if stage == 'exp10_sensitivity':
                sens = cfg.get('monitor_sensitivity', {})
                sens_jobs = int(sens.get('jobs', runtime.get('exp10_sensitivity_jobs', jobs)))
                extra_args.extend(['--jobs', str(max(1, sens_jobs))])
                if sens.get('batch_sizes'):
                    extra_args.extend(['--batch-sizes', ','.join(str(x) for x in sens['batch_sizes'])])
                if sens.get('ref_sizes'):
                    extra_args.extend(['--ref-sizes', ','.join(str(x) for x in sens['ref_sizes'])])
                if sens.get('repeats') is not None:
                    extra_args.extend(['--repeats', str(sens['repeats'])])
            if stage == 'exp_knn_sensitivity':
                knn_jobs = int(runtime.get('exp_knn_sensitivity_jobs', jobs))
                extra_args.extend(['--jobs', str(max(1, knn_jobs))])
            if stage == 'exp_competitor_switching':
                comp_jobs = int(runtime.get('exp_competitor_switching_jobs', jobs))
                extra_args.extend(['--jobs', str(max(1, comp_jobs))])
            run_module(module, extra_args)

        mark_done(cfg, state, stage)

    elapsed = (time.time() - t0) / 3600
    print(f'\nFull run finished in {elapsed:.2f} hours.')
    print(f'Results: {cfg["paths"]["results_dir"]}')


if __name__ == '__main__':
    main()
