def run_2_1():
    # %%
    # TODO need to rerun pgd and shgo based optimizer due to now always using adam due to scale invariance
    # %%
    # TODO technical the per class knn is set to 1 neighbor which is not the deault. Mention this in thesis even though results might be slighlty differetnt should not be a big difference.
    # %%
    # TODO technical speaking rerun psa for coil 100
    # %%
    import copy

    import torch
    import torch.nn as nn
    import torchvision
    import numpy as np
    from matplotlib import pyplot as plt

    from its.search import InverseTransformationSearch
    from search.parallel_gradient import ParallelGradientDescent
    from utils.affine_transforms_old import AffineTransformation2D
    from utils.sampling import BatchNegativeSampler

    # torch.cuda.is_available = lambda: False
    # device = torch.device("cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # look for experiment files in parents
    import os

    path_found = False
    current_path = os.getcwd()
    while not path_found:
        if os.path.exists(os.path.join(current_path, "experiment_files")):
            path_found = True
            break
        current_path = os.path.dirname(current_path)

    experiment_files_path_data = os.path.join(current_path, "experiment_files", "data")
    dataset = "modelnet10"

    default_architecutre_mapping = {
        "mnist": "resnet_small",
        "bigger_mnist": "resnet_small",
        "emnist": "extended_resnet_small",
        "bigger_emnist": "bigger_extended_resnet_small",
        "coil100": "coil_resnet_small",
        "tu_berlin": "bi_lstm",
        "modelnet10": "pointnetplus",

    }

    architecture = default_architecutre_mapping[dataset]
    budget = None
    # %%
    from utils.transforms.apply import grid_resample_border, grid_resample_reflection
    # %%

    # %%
    from experiment_thesis.dataset_preperation.get_dataset import get_dataset_info, get_dataset

    dataset_info = get_dataset_info(dataset)

    dataset_dict = get_dataset(dataset_info, path=experiment_files_path_data, batch_size=dataset_info.batch_size)
    transform_name = dataset_info.transform_seq_name
    # %%

    # %%

    # %%

    # %%

    # %%

    dataset_dict.keys()
    dataset_train = dataset_dict['train_dataset']
    dataset_val = dataset_dict['val_dataset']
    dataset_test = dataset_dict['test_dataset']
    train_loader = dataset_dict['train_loader']
    val_loader = dataset_dict['val_loader']
    test_loader = dataset_dict['test_loader']
    n_classes = dataset_info.num_classes
    train_loader_transformed = dataset_dict['train_loader_transformed']
    val_loader_transformed = dataset_dict['val_loader_transformed']
    test_loader_transformed = dataset_dict['test_loader_transformed']
    train_loader_no_shuffle = dataset_dict['train_loader_no_shuffle']
    # %%
    x = next(iter(test_loader_transformed))[0]
    # %%

    # %%
    batch_size = next(iter(train_loader))[0].shape[0]

    # %%
    from utils.eval.vis import vis_dataset

    vis_dataset(train_loader, val_loader, test_loader_transformed)
    # %%
    from experiment_thesis.main import train_and_get_model, train_or_load_energy_model
    from experiment_thesis.dataset_preperation.basic_networks import get_network
    from utils.eval.main_model import evaluate_base_model

    model_dir_path = os.path.join(current_path, "experiment_files", "models")
    embedding_cache_path = os.path.join(current_path, "experiment_files", "embedding_cache")
    # Add results dir and helper for save paths
    results_dir_path = os.path.join(current_path, "experiment_files", "results", dataset, architecture,
                                    "comparision_over_budget")
    os.makedirs(results_dir_path, exist_ok=True)

    def savepath(label: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
        return os.path.join(results_dir_path, transform_name, f"{safe}.json")

    # %%
    model = get_network(dataset_info, architecture, num_classes=n_classes).to(device)
    modelname = f"{dataset}_{architecture}"
    cache_name_train = f"{dataset}_{architecture}_embedding_cache_train"

    train_and_get_model(model, model_dir_path, modelname, train_loader, val_loader, trainer_kwargs={
        "accelerator": "auto",
        "max_epochs": dataset_info.epochs,
        "precision": "16-mixed",
    }, load_if_exists=True)

    # %%
    model.eval().to(device)
    # %%
    res = evaluate_base_model(model, test_loader_transformed, device)
    print(res)
    # %%
    res = evaluate_base_model(model, test_loader, device)
    print(res)
    # %%
    res = evaluate_base_model(model, val_loader, device)
    print(res)

    # %%
    class TensorGeometricModelUnwrapper(torch.nn.Module):
        """
        Wrapper for a torch_geometric model that receives a tuple of (pos, y) as input and creates
        a Data object from it that is passed to the model.
        """

        def __init__(self):
            super(TensorGeometricModelUnwrapper, self).__init__()

        def forward(self, data):
            # pos and y are batched tensors from a DataLoader
            # need to reconstruct the original Data objects for torch_geometric models

            pos = data.pos
            batch = data.batch
            # split pos into individual tensors based on batch
            pos_list = torch.split(pos, torch.bincount(batch).tolist())
            return torch.stack(pos_list)

    # %%
    # chek if data is iamge data
    is_image_data = len(dataset_info.input_size) == 3 and dataset_info.input_size[0] in [1, 3]
    # %%
    from utils.transforms.apply import grid_resample
    from experiment_thesis.dataset_preperation.transformation import get_transformation_sequence_images

    transform_seq = get_transformation_sequence_images(
        name=dataset_info.transform_seq_name,
        resample_method=dataset_info.resample_method,
        init_method="sobol"
    ).to(device)
    # %%
    from experiment_thesis.dataset_preperation.basic_networks import get_network_layer

    layer, layer_io = get_network_layer(dataset_info, architecture, 0, num_classes=None, num_rotations=8)
    # %%
    from confidence.direct.logit_based import EnergyConfidence
    from utils.transformation_problem import TransformationProblem
    from confidence.model.single_pass import SinglePassConfidence

    logit_energy = SinglePassConfidence(model, EnergyConfidence(), index=None)
    problem_energy_logits = TransformationProblem(logit_energy, transform_seq, consolidate_method="consolidate_simple")
    # test ot
    from search.shgo import SHGO
    random_search = SHGO(initial_samples=120, local_max_steps=0)

    from utils.eval.ood_performance import load_or_run_evaluate_confidence_and_search, evaluate_confidence_and_search, \
        ITSWRAPPER

    # %%
    load_or_run_evaluate_confidence_and_search(
        model, optimizer=random_search, problem=problem_energy_logits,
        test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
        save_path=savepath("energy_confidence_transformed"), show_progress=True,
        repeats=1)
    # %%
    model.to(device).eval()
    # %%
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # %%

    # %%

    # %%

    # %%
    if "modelnet10" in dataset:
        for init_method in ["individual", "sobol"]:
            print(f"Testing init method: {init_method}")
            import torch
            import kornia
            import matplotlib.pyplot as plt
            import numpy as np
            from mpl_toolkits.mplot3d import Axes3D
            from utils.affine_transforms import AffineTransformations3D
            from utils.transform_sequence import TransformSequence
            from scipy import stats

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Save original init method and temporarily switch to "individual"
            original_init = transform_seq.init_method
            transform_seq.init_method = init_method

            # --- 1. Generate your matrices ---
            zw = transform_seq.initial_param(1280)
            transform_seq.init_method = original_init
            matrix = transform_seq(zw)

            # --- 2. Extract rotation submatrices ---
            R = matrix[:, :3, :3]  # [N, 3, 3]

            # --- 3. Convert to axis–angle representation ---
            rotvec = kornia.geometry.conversions.rotation_matrix_to_axis_angle(R)  # [N, 3]
            theta = torch.linalg.norm(rotvec, dim=1)
            axis = torch.nn.functional.normalize(rotvec, dim=1)

            axis_np = axis.cpu().numpy()
            theta_np = theta.cpu().numpy()

            # --- 4. Quantitative checks ---

            # (a) Axis uniformity
            mean_axis = axis_np.mean(axis=0)
            spherical_var = 1 - np.linalg.norm(mean_axis)
            print("Axis mean vector:", mean_axis)
            print("Spherical variance (1 - |mean|):", spherical_var)

            # (b) Rotation angle
            sample_mean = np.mean(theta_np)
            theory_mean = np.pi / 2 + 2 / np.pi  # Correct expected mean for uniform SO(3)
            print(f"Sample mean rotation angle (rad): {sample_mean:.4f}")
            print(f"Theoretical mean rotation angle (rad): {theory_mean:.4f}")

            # (c) Kolmogorov–Smirnov test against theoretical CDF
            def F_theta(t):
                # CDF for p(theta) ∝ sin²(theta/2)
                return (t - np.sin(t)) / np.pi

            ks_stat, pval = stats.kstest(theta_np, F_theta)
            print(f"KS statistic: {ks_stat:.4f}, p-value: {pval:.4f}")

            # --- 5. Visualization ---

            fig = plt.figure(figsize=(12, 5))

            # Angle histogram
            plt.subplot(1, 2, 1)
            plt.hist(theta_np, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black')
            t_grid = np.linspace(0, np.pi, 200)
            plt.plot(t_grid, (2 / np.pi) * np.sin(t_grid / 2) ** 2, 'r-', lw=2, label='theoretical pdf')
            plt.xlabel("Rotation angle (radians)")
            plt.ylabel("Density")
            plt.title("Rotation Angle Distribution")
            plt.legend()

            # 3D scatter of axes
            ax = fig.add_subplot(1, 2, 2, projection='3d')
            ax.scatter(axis_np[:, 0], axis_np[:, 1], axis_np[:, 2], s=8, alpha=0.7)
            # Draw reference sphere
            u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
            x = np.cos(u) * np.sin(v)
            y = np.sin(u) * np.sin(v)
            z = np.cos(v)
            ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)
            ax.set_box_aspect([1, 1, 1])
            ax.set_title("Rotation Axes on Unit Sphere")

            plt.tight_layout()
            plt.show()

    # %%

    # %%
    from utils.augments import ComposeAugmentations, random_gaussian_noise, random_contrast, \
        random_gamma, random_blur_or_sharpen, build_default_augmentations
    import utils.augments

    def dec_strat(x, idd, y_true):
        out = model(x)
        eq = out.argmax(dim=-1) == y_true
        # convert to tensor where y>=0 if correct, y<0 if incorrect
        y = torch.where(eq, y_true, -1)
        return y

    from utils.augments import build_default_augmentations, small_affine_augment_2d
    from utils.sampling_strategy import GaussianSamplingStrategyLatent, TransformLatentSamplingStrategy
    import importlib
    import utils.sampling_strategy
    import utils.sampling

    importlib.reload(utils.sampling)
    from utils.sampling import BatchNegativeSampler

    energy_model2 = get_network(dataset_info, architecture, num_classes=1).to(device)

    from experiment_thesis.main import train_or_load_energy_model

    if is_image_data:
        transform_true_function = small_affine_augment_2d
        affine_augment = utils.augments.build_default_augmentations()
    else:
        transform_true_function = None
        affine_augment = None

    negative_sampling_module = BatchNegativeSampler(
        TransformLatentSamplingStrategy(
            transform_sequence=transform_seq, ), transform_true_function
        =transform_true_function, augment_function=affine_augment,
        decision_strategy=dec_strat,
    )

    energy_conf2 = train_or_load_energy_model(
        energy_model2, model_dir_path, f"{modelname}_energy2", train_loader,
        val_loader, trainer_kwargs={
            "accelerator": "auto",
            "max_epochs": dataset_info.epochs // 2,
            "precision": "16-mixed" if dataset_info.name not in ["modelnet10"] else "32",
        }, negative_sampling_module=negative_sampling_module, load_if_exists=True)

    # %%
    model.to(device).eval()
    # %%
    from model.pointnet_plus import SAModule
    def set_deterministic_fps(model, random_start=False):
        for module in model.modules():
            if isinstance(module, SAModule):
                module.random_start = random_start
                print(f"Set random_start={random_start} for {module.__class__.__name__}")

    set_deterministic_fps(model)
    set_deterministic_fps(energy_model2)
    # %%
    energy_conf2.to(device).eval()

    problem_energy2 = TransformationProblem(energy_conf2, transform_seq, consolidate_method="consolidate_simple")

    # %%
    load_or_run_evaluate_confidence_and_search(
        model, optimizer=random_search, problem=problem_energy2,
        test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
        save_path=savepath("learned_energy_confidence_transformed"), show_progress=True,
        repeats=1)
    # %%
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # %%

    # %%
    from torch.utils.data import SequentialSampler
    from embedding_cache import LayerEmbeddingCache

    cache_name_train = f"{dataset}_{architecture}_{transform_name}_embedding_cache_train"

    cache_train = LayerEmbeddingCache(model, train_loader_no_shuffle,
                                      cache_dir=os.path.join(embedding_cache_path, cache_name_train))

    dual_output_model = cache_train.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True)
    embeddings_t, final_t, classes_t = cache_train.__call__(layer, capture_modes=layer_io, flatten=True)

    from utils.transformation_problem import TransformationProblem
    from confidence.model.single_pass import SinglePassConfidence
    from confidence.direct.logit_based import EnergyConfidence
    from confidence.control.split import SplitConfidence, PredictedSplitConfidence
    from confidence.unsupervised.classic.nn_pytorch import KNNConfidence, PerClassKNNConfidence

    from confidence.input_transform import InputTransformImage, PCAInputModule, RandomProjectionModule

    nn_pytorch_pretrained = PerClassKNNConfidence(metric="cosine", input_transform=None, computation_mode="masked", k=1)
    nn_pytorch_pretrained.fit(embeddings_t, classes_t)
    nn_pytorch_pretrained.to(device)

    conf_split_pretrained = PredictedSplitConfidence(nn_pytorch_pretrained, EnergyConfidence(), mult=False, b=0.0)
    conf_mod_nn_pytorch_pretrained = SinglePassConfidence(dual_output_model, conf_split_pretrained, index=1)
    problem_nn_pytorch_pretrained = TransformationProblem(conf_mod_nn_pytorch_pretrained, transform_seq,
                                                          consolidate_method="consolidate_simple")
    model.eval().to(device)
    # %%

    # %%
    # benchmark model and dual output model
    # %%
    load_or_run_evaluate_confidence_and_search(
        model, optimizer=random_search, problem=problem_nn_pytorch_pretrained,
        test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
        save_path=savepath("knn_per_class_confidence_transformed"), show_progress=True,
        repeats=1, overwrite=False)
    # %%

    # %%

    # %%

    # %%
    gc.collect()
    torch.cuda.empty_cache()
    # %%
    x = next(iter(train_loader_no_shuffle))

    res1 = dual_output_model(x[0].to(device))[0].cpu().detach().numpy()

    res2 = embeddings_t[:x[0].shape[0]].cpu().detach().numpy()
    if not np.allclose(res1, res2):
        pass
        # raise ValueError("Model is not deterministic!")
    else:
        print("Model is deterministic.")

    del res1
    del res2
    del x
    # %%
    from utils.eval.ood_performance import ITSWRAPPER
    import importlib
    import its.search
    importlib.reload(its.search)

    its2 = ITSWRAPPER(its.search.InverseTransformationSearch(model, None, None, n_hypotheses=1, n_samples=10, extend=0,
                                                             gaussian_filter_channel_wise=True))
    # %%
    from search.tree import CoordinateDescent
    cd = CoordinateDescent()
    # %%
    gc.collect(

    )
    torch.cuda.empty_cache()
    # %%

    # %%

    # %%
    gc.collect()
    torch.cuda.empty_cache()
    # %%
    x = next(iter(test_loader_transformed))[0].to(device)
    # %%

    # %%

    # %%

    # %%
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # %%

    # %%

    # %%

    # %%
    from utils.replacer import replace_rotation_transforms, replace_rotation_transforms_2vec
    import os
    import optuna
    import torch
    import gc

    from search.objective_generators import (
        make_search_objective,
        save_best_trial_params,
        build_search_algorithm,
        _cost_shgo,  # cost helper for sanity checks
    )
    from search.config import load_params, get_default_params, save_params
    from utils.eval.ood_performance import load_or_run_evaluate_confidence_and_search

    model.eval().to(device)
    # detach model for efficent gradients
    for param in model.parameters():
        param.requires_grad = False

    for param in energy_model2.parameters():
        param.requires_grad = False

    # Algorithms and budgets
    all_algos = ["shgo", "parallel_sa",
                 "evolutionary", "pso", "cd", "wcd",
                 "pgd", "pgd_restart", "pgd_window", "its", "random_search", "cd_multi_cyclus", "wcd_lattice", "its2"
                 # "cmaes" #cmaes seems currently alway to default to only using 0 iterations.
                 # add "its" if you have an ITS optimizer/module wired up
                 ]

    # Add complex/quaternion variants
    complex_algos = ["shgo_c", "pgd_c", "shgo_c_3"]  # NEW: added shgo_c_3 and pgd_c_3
    all_algos.extend(complex_algos)

    if dataset == "modelnet10":
        all_algos.append("shgo_individual")
        all_algos.append("wcd_lat_ind")
        all_algos.append("pgd_vector")
        all_algos.append("shgo_vector")

    # remove pgd pgd restarts and pgd window size for tu berlin
    if dataset in ["tu_berlin"]:
        all_algos = [
            "its2", "shgo", "parallel_sa",
            "evolutionary", "pso", "cd", "wcd", "its", "random_search", "wcd_lattice", "cd_multi_cyclus"]
        grad_weight = 99999999
    else:
        grad_weight = 2

    budgets = [60]

    grad_weight_algos = {"shgo", "shgo_individual", "pgd", "pgd_restart", "pgd_window",
                         "shgo_c", "pgd_c", "pgd_restart_c", "shgo_c_3", "pgd_c_3", "pgd_vector", "shgo_vector"}
    default_trials = 30
    eval_repeats = 4
    if dataset in ["coil100", "tu_berlin", "modelnet10"]:
        eval_repeats = 4

    show_progress = True

    # Problem configurations
    problems = [problem_energy_logits, problem_nn_pytorch_pretrained, problem_energy2]
    problem_names = ["logit_energy", "knn_per_class", "learned_energy"]

    # Create complex/quaternion versions of problems
    problems_complex = [replace_rotation_transforms(p) for p in problems]
    problem_names_complex = problem_names

    # New problems for shgo_individual
    problems_individual = [ITSWRAPPER._prepare_problem(p) for p in problems]
    problem_names_individual = problem_names

    problems_rotvec = [replace_rotation_transforms_2vec(p) for p in problems]
    problem_names_rotvec = problem_names

    for p in problems:
        p.max_batch_size = dataset_info.batch_size_search
    for p in problems_individual:
        p.max_batch_size = dataset_info.batch_size_search
    for p in problems_complex:
        p.max_batch_size = dataset_info.batch_size_search

    # Base results directory
    base_results_dir = os.path.join(
        current_path, "experiment_files", "search_results",
        str(dataset), dataset_info.transform_seq_name, str(architecture)
    )
    os.makedirs(base_results_dir, exist_ok=True)

    assert len(problems) == len(problem_names), "Mismatch between problems and problem_names."

    # Mapping from complex variants to their base algorithms
    algo_variant_mapping = {
        "pgd_c": "pgd",
        "shgo_c": "shgo",
        "pgd_restart_c": "pgd_restart",
        "shgo_c_3": "shgo",  # NEW: map to base shgo
        "pgd_c_3": "pgd",  # NEW: map to base pgd
        "pgd_vector": "pgd",
        "shgo_vector": "shgo",
    }
    grad_weight_orig = grad_weight
    for budget in budgets:
        print(f"\n=== Budget: {budget} ===")
        for algo in all_algos:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"\n--- Algorithm: {algo} ---")

            # NEW: special grad_weight override
            if algo == "shgo_c_3" or algo == "pgd_c_3":
                grad_weight = 3
            else:
                grad_weight = grad_weight_orig

            # Determine which problems to use
            if algo in complex_algos:
                current_problems = problems_complex
                current_problem_names = problem_names_complex
            elif algo in ["shgo_individual", "wcd_lat_ind", "its", "its2"]:
                current_problems = problems_individual
                current_problem_names = problem_names_individual
            elif algo in ["pgd_vector", "shgo_vector"]:
                current_problems = problems_rotvec
                current_problem_names = problem_names_rotvec
            else:
                current_problems = problems
                current_problem_names = problem_names

            for p in current_problems:
                p.max_batch_size = dataset_info.batch_size_search

            # Map algorithm names for construction
            if algo in complex_algos:
                algo_name_for_path = algo_variant_mapping[algo]
            elif algo == "shgo_individual":
                algo_name_for_path = "shgo"
            elif algo == "wcd_lat_ind":
                algo_name_for_path = "wcd_lattice"
            elif algo in ["pgd_vector", "shgo_vector"]:
                algo_name_for_path = algo_variant_mapping[algo]
            else:
                algo_name_for_path = algo

            algo_dir = os.path.join(base_results_dir, algo, f"budget_{budget}")
            os.makedirs(algo_dir, exist_ok=True)

            param_path = os.path.join(algo_dir, "best.yml")
            print(f"Result directory: {algo_dir}")

            # Load stored params or optimize
            stored_params = load_params(param_path) if os.path.exists(param_path) else None
            if stored_params is None and (algo != "cd" and "random_search" not in algo):
                default_params_kwargs = {}
                if algo in grad_weight_algos:
                    default_params_kwargs["grad_weight"] = grad_weight
                default_params = get_default_params(algo_name_for_path, budget, **default_params_kwargs)
                print("Default params (config):", default_params)

                objective_kwargs = {}
                if algo in grad_weight_algos:
                    objective_kwargs["grad_weight"] = grad_weight

                objective = make_search_objective(
                    algo=algo_name_for_path,
                    model=model,
                    val_loader=val_loader_transformed,
                    problem=current_problems,  # multi-problem objective
                    budget=budget,
                    device=str(device),
                    repeats=1,
                    **objective_kwargs,
                )

                pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
                study = optuna.create_study(direction="maximize", pruner=pruner)
                study.enqueue_trial(default_params)

                study.optimize(objective, n_trials=default_trials, show_progress_bar=False)

                print(f"[{algo}] Best validation value:", study.best_value)
                print("Suggested params:", study.best_trial.params)
                print("Full params:", study.best_trial.user_attrs.get("full_params"))

                save_best_trial_params(study, algo=algo_name_for_path, path=param_path)
                stored_params = load_params(param_path)
                print("Saved best params to:", param_path)
            else:
                print("Found stored best params, skipping optimization.")
                if "cd" == algo:
                    print("Note: 'cd' algorithm requires no hyperparameter optimization.")
                    stored_params = {}

            # SHGO-only sanity check: force full budget consumption by topping up n_init
            if algo in ["shgo", "shgo_individual", "shgo_c", "shgo_c_3"]:  # NEW: added shgo_c_3
                gw = stored_params.get("grad_weight", 2)
                cost = _cost_shgo(
                    stored_params["shgo_initial_samples"],
                    stored_params["shgo_local_runs"],
                    stored_params["shgo_local_steps"],
                    gw,
                )
                if cost != budget:
                    delta = budget - cost
                    # Adjust n_init up/down to match budget exactly
                    stored_params["shgo_initial_samples"] = max(
                        1,
                        stored_params["shgo_initial_samples"] + delta
                    )
                    # Recompute and assert
                    new_cost = _cost_shgo(
                        stored_params["shgo_initial_samples"],
                        stored_params["shgo_local_runs"],
                        stored_params["shgo_local_steps"],
                        gw,
                    )
                    print(f"[{algo}] adjusted n_init by {delta} to match budget: {cost} -> {new_cost}")
                    assert new_cost == budget, f"SHGO cost mismatch after fix: {new_cost}!={budget}"
                    # Persist the fix
                    save_params(stored_params, param_path)

            # assert that grad weight matches
            if algo in grad_weight_algos:
                assert stored_params.get("grad_weight", None) == grad_weight, \
                    f"Grad weight mismatch for {algo}: {stored_params.get('grad_weight', None)}!={grad_weight}"

            # Rebuild optimizer with best params
            search_obj = build_search_algorithm(
                algo_name_for_path,
                stored_params,
                problem=current_problems[0],  # any problem is fine for optimizer construction
                budget=budget,
                model=model,
            )
            print("Rebuilt search object from saved params.")

            # Evaluate per problem with cached runner
            for prob, method_name in zip(current_problems, current_problem_names):
                eval_path = os.path.join(algo_dir, f"eval_{method_name}.json")
                print(f"[{method_name}] evaluating (cached path: {eval_path})")
                metrics = load_or_run_evaluate_confidence_and_search(
                    model=model,
                    optimizer=search_obj,
                    problem=prob,
                    test_loader=test_loader_transformed,
                    save_path=eval_path,  # auto-loads if exists; saves after run otherwise
                    max_batch_override=dataset_info.batch_size_search,
                    show_progress=show_progress,
                    repeats=eval_repeats,
                    return_per_run=True,
                    overwrite=False, store_val=True
                )
                gc.collect()
                torch.cuda.empty_cache()
    # %%

    # %%

    # %%
    import pandas as pd
    import json
    import os
    import matplotlib.pyplot as plt

    # Collect results
    results = []

    for budget in budgets:
        for algo in all_algos:
            algo_dir = os.path.join(base_results_dir, algo, f"budget_{budget}")
            for method_name in problem_names:
                eval_path = os.path.join(algo_dir, f"eval_{method_name}.json")
                if os.path.exists(eval_path):
                    with open(eval_path, "r") as f:
                        metrics = json.load(f)

                    accuracy = metrics.get("accuracy_mean", None)
                    accuracy_se = metrics.get("accuracy_se", None)
                    accuracy_std = metrics.get("accuracy_std", None)
                    number_of_runs = metrics.get("repeats", None)
                    results.append({
                        "Algorithm": algo,
                        "Budget": budget,
                        "Problem": method_name,
                        "Accuracy": accuracy,
                        "Accuracy_SE": accuracy_se,
                        "Accuracy_STD": accuracy_std,
                        "Number_of_Runs": number_of_runs,

                    })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # %%
    df
    # %%
    transform_seq.transformations
    # %%
    ALGO_RENAME = {
        "cd": "CD-single",
        "cd_multi_cyclus": "CD",
        "shgo": "RS-LO",
        "shgo_individual": "RS-LO-Ind",
        "parallel_sa": "PSA",
        "evolutionary": "Evo. A",
        "pso": "PSO",
        "pgd": "PGD",
        "pgd_restart": "PGD-R",
        "pgd_window": "PGD-W",
        "its": "ITS",
        "random_search": "R. Search",
        "wcd": "WCD",
        "wcd_lattice": "WCD-Lat",
        "shgo_c": "RS-LO-C",
        "pgd_c": "PGD-C",
        "shgo_c_3": "RS-LO-C3",
        "pgd_c_3": "PGD-C3",
        "wcd_lat_ind": "WCD-Lat-Ind",
        "pgd_vector": "PGDVec",
        "shgo_vector": "RS-LO-Vec",
    }

    PROBLEM_RENAME = {
        "knn_per_class": "PC-kNN",
        "logit_energy": "Logit-Energy",
        "learned_energy": "Learned-Energy",
    }

    # %%
    # do some renaming

    df_renamed = df.copy()
    # rename cd to cd_signle_cycle
    # rename cd_multi_cyclus to cd
    # rename shgo to RS-LO
    df_renamed["Algorithm"] = df["Algorithm"].replace(ALGO_RENAME)

    # reanme problem name from knn_per_class to PC-kNN
    df_renamed["Problem"] = df["Problem"].replace(PROBLEM_RENAME)
    problem_names_renamed = ["Logit-Energy", "PC-kNN", "Learned-Energy"]

    # %%

    # %%
    fullname_dict = {
        "CD-single": "Coordinate Descent (single cycle)",
        "CD": "Coordinate Descent",
        "RS-LO": "Random Sampling with Local Optimization",
        "ITS": "Inverse Transformation Search",
        "R. Search": "Random Search",
        "PSA": "Parallel Simulated Annealing",
        "Evo. A": "Evolutionary Algorithm",
        "PSO": "Particle Swarm Optimization",
        "PGD": "Multistart Gradient Descent",
        "PGD-R": "Multistart Gradient Descent with Restarts",
        "PGD-W": "Multistart Gradient Descent with Windowing",
        "WCD": "Weighted Coordinate Descent",
        "WCD-Lat": "Weighted Coordinate Descent with Lattice Sampling",
        "RS-LO-Ind": "Random Sampling with Local Optimization (individual rotations)",
        "RS-LO-C": "Random Sampling with Local Optimization (complex/quaternion rotations)",
        "PGD-C": "Multistart Gradient Descent (complex/quaternion rotations)",
        "RS-LO-C3": "Random Sampling with Local Optimization complex/quaternion rotations, grad weight=3",
        "PGD-C3": "Multistart Gradient Descent complex/quaternion rotations, grad weight=3",
        "WCD-Lat-Ind": "Weighted Coordinate Descent with Lattice Sampling (euler rotations)",
        "PGDVec": "Multistart Gradient Descent (rotation vector representation)",
        "RS-LO-Vec": "Random Sampling with Local Optimization (rotation vector representation)",
    }

    # %%

    # %%
    import matplotlib.pyplot as plt

    short_names = list(fullname_dict.keys())
    n = len(short_names)

    # Get all 20 colors from tab20
    tab20_colors = [plt.get_cmap("tab20")(i)[:3] for i in range(20)]

    # Separate even and odd indices
    even_colors = tab20_colors[::2]  # 0, 2, 4, ..., 18
    odd_colors = tab20_colors[1::2]  # 1, 3, 5, ..., 19

    # Combine: even first, then odd
    ordered_colors = even_colors + odd_colors

    # Repeat pattern if more names than 20
    if n > 20:
        repeats = (n // 20) + 1
        ordered_colors = (ordered_colors * repeats)[:n]
    else:
        ordered_colors = ordered_colors[:n]

    # Assign to names
    algorithm_colors = {name: ordered_colors[i] for i, name in enumerate(short_names)}

    # %%
    figure_path = os.path.join(current_path, "experiment_files", "export", "fig", "comparision_search_datasets",
                               dataset, transform_name)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    # %%
    from matplotlib.patches import Patch
    from utils.eval.vis import plt_setup_latex
    W = plt_setup_latex()
    handles = [
        Patch(color=algorithm_colors[short], label=f"{short} — {fullname_dict[short]}")
        for short in fullname_dict
    ]

    # Plot legend in a separate figure
    fig, ax = plt.subplots(figsize=(W, W * 0.4))
    ax.legend(
        handles=handles,
        loc='center',
        frameon=False,
        fontsize=9
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, "comparision_search_algorithms_legend.pdf"))
    plt.savefig(os.path.join(figure_path, "comparision_search_algorithms_legend.pgf"))
    plt.show()
    # %%

    # %%
    from utils.eval.vis import plt_setup_latex

    W = plt_setup_latex()
    # %%
    df_renamed["Accuracy_SE"] = pd.to_numeric(df_renamed["Accuracy_SE"], errors="coerce").fillna(0)

    fig, axes = plt.subplots(
        1, len(problem_names_renamed),
        figsize=(W * 2, W / 1.5),
        constrained_layout=True
    )
    axes = axes if len(problem_names_renamed) > 1 else [axes]

    for idx, (ax, problem) in enumerate(zip(axes, problem_names_renamed)):
        sub = df_renamed[df_renamed["Problem"] == problem].sort_values("Algorithm")
        algos = sub["Algorithm"].tolist()
        accs = sub["Accuracy"].to_numpy()
        ses = sub["Accuracy_SE"].to_numpy()

        x = np.arange(len(algos))

        # Directly map colors using algorithm_colors dict
        bar_colors = [algorithm_colors.get(a, (0.5, 0.5, 0.5)) for a in algos]

        ax.bar(
            x, accs, yerr=ses, capsize=2,
            alpha=0.85, ecolor='black', error_kw={'elinewidth': 0.8},
            color=None
        )

        ymin = max(0, accs.min() - 0.05)
        ymax = min(1, accs.max() + 0.05)
        ax.set_ylim(ymin, ymax)

        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=45, ha='right', fontsize=6)
        ax.set_title(problem)

        if idx == 0:
            ax.set_ylabel("Accuracy")
        ax.set_xlabel("Algorithm")
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
    # %%

    # %%

    # %%
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    def plot_accuracy_bars(
            df,
            problem_column="Problem",
            algo_column="Algorithm",
            mean_column="Accuracy",
            se_column="Accuracy_SE",
            problem_names=None,
            algorithm_colors=None,
            figure_path=None,
            fig_width=None,
            fig_height=None,
            error_capsize=1.5,
            save_name="comparison_search_algorithms"
    ):
        """
        Plots horizontal bar charts of accuracy with error bars for each problem.
        Works for both individual problem scores and grouped scores.

        Parameters:
            df : pd.DataFrame
                Data containing algorithm performance.
            problem_column : str
                Column name for problems.
            algo_column : str
                Column name for algorithms.
            mean_column : str
                Column name for mean accuracy.
            se_column : str
                Column name for standard error.
            problem_names : list
                Ordered list of problems to plot. If None, all unique problems in df are used.
            algorithm_colors : dict
                Mapping from algorithm name to color (RGB tuple or string). Defaults to grey if missing.
            figure_path : str
                Directory to save the figure. If None, figure is not saved.
            fig_width, fig_height : float
                Figure dimensions. If None, defaults are computed automatically.
            error_capsize : float
                Capsize for error bars.
            save_name : str
                Base name for saved figure files (PDF & PGF).
        """
        plt.style.use("seaborn-v0_8-whitegrid")

        if problem_names is None:
            problem_names = df[problem_column].unique()

        if algorithm_colors is None:
            algorithm_colors = {}

        n_problems = len(problem_names)

        if fig_width is None:
            fig_width = 8
        if fig_height is None:
            fig_height = fig_width / max(2.2, n_problems / 3)

        fig, axes = plt.subplots(
            1, n_problems,
            figsize=(fig_width, fig_height),
            sharey=False
        )

        if n_problems == 1:
            axes = [axes]

        for idx, (ax, problem) in enumerate(zip(axes, problem_names)):
            sub = df[df[problem_column] == problem].sort_values(algo_column)

            algos = sub[algo_column].tolist()
            accs = sub[mean_column].to_numpy()
            ses = sub[se_column].to_numpy()
            y = np.arange(len(algos))

            colors = [algorithm_colors.get(a, (0.5, 0.5, 0.5)) for a in algos]

            ax.barh(
                y, accs, xerr=ses,
                capsize=error_capsize,
                color=colors,
                edgecolor="none",
                height=0.7,
                alpha=1,
                error_kw={'elinewidth': 1.5, 'ecolor': 'black', 'capsize': 1}
            )

            ax.set_yticks(y)
            if idx == 0:
                ax.set_yticklabels(algos, fontsize=6)
                ax.set_ylabel("Algorithm", fontsize=10)
            else:
                ax.set_yticklabels([])

            ax.invert_yaxis()  # first algorithm on top

            xmin = max(0, (accs - ses).min() - 0.03)
            xmax = min(1, (accs + ses).max() + 0.03)
            ax.set_xlim(xmin, xmax)

            ax.set_xlabel("Accuracy", fontsize=10)
            ax.set_title(problem, fontsize=11)
            ax.grid(True, axis='x', linestyle='--', alpha=0.5)
            sns.despine(ax=ax, left=True, bottom=False)

        plt.tight_layout(pad=0.5, w_pad=0.2)

        if figure_path is not None:
            os.makedirs(figure_path, exist_ok=True)
            plt.savefig(os.path.join(figure_path, f"{save_name}.pdf"))
            plt.savefig(os.path.join(figure_path, f"{save_name}.pgf"))

        plt.show()

    plot_accuracy_bars(
        df=df_renamed,
        problem_names=problem_names_renamed,
        algorithm_colors=algorithm_colors,
        figure_path=figure_path
    )

    # %%
    import numpy as np
    import pandas as pd

    def compute_grouped_scores(group):
        """
        Compute combined mean and SE across problems.
        """
        N = len(group)  # number of problems
        # Mean across problems
        mean_accuracy = group["Accuracy"].mean()

        # Variance of each problem mean: s_i^2 / n_i
        var_each = (group["Accuracy_STD"].to_numpy() ** 2) / group["Number_of_Runs"].to_numpy()

        # Combined SE across problems
        se_combined = np.sqrt(np.sum(var_each)) / N

        return pd.Series({
            "Grouped_Accuracy_Mean": mean_accuracy,
            "Grouped_Accuracy_SE": se_combined
        })

    # Apply grouping
    grouped_scores = df_renamed.groupby(["Algorithm", "Budget"]).apply(compute_grouped_scores).reset_index()
    grouped_scores["Problem"] = "Grouped"
    print(grouped_scores)

    # plot grouped scores
    # %%
    plot_accuracy_bars(
        df=grouped_scores,  # DataFrame with Grouped_Accuracy_Mean & Grouped_Accuracy_SE
        mean_column="Grouped_Accuracy_Mean",
        se_column="Grouped_Accuracy_SE",
        problem_names=None,  # will use all problems if None
        algorithm_colors=algorithm_colors,
        figure_path=figure_path,
        save_name="grouped_comparison_search_algorithms",
        fig_width=W / 2,
        fig_height=W / 2.2,
    )

    # %%

    # %%

    # %%
    # plot grouped accuracy across problems

    # %%

    # %%

    # %%

    # %%
    def analyze_run_results(results_list):
        """
        Computes summary statistics per algorithm for each problem.
        Uses all runs per algorithm.
        Normalizes errors per sample using min–max scaling across all runs & algorithms.
        Frobenius distance is computed against the best run per sample (lowest raw error).
        Checks for consistency of true labels and includes predicted labels from the best run.
        """
        import numpy as np
        from collections import defaultdict

        results_by_problem = defaultdict(list)
        for entry in results_list:
            results_by_problem[entry["Problem"]].append(entry)

        problem_summaries = {}

        for problem, entries in results_by_problem.items():
            # Collect all errors, matrices, and labels
            all_errors, all_mats, algo_labels = [], [], []
            all_true_labels, all_pred_labels = [], []
            run_metadata = []

            for entry in entries:
                algo = ALGO_RENAME.get(entry["Algorithm"], entry["Algorithm"])
                problem_name = PROBLEM_RENAME.get(entry["Problem"], entry["Problem"])

                entry["Algorithm"] = algo
                entry["Problem"] = problem_name

                res = entry["metrics"]
                runs = res.get("per_run", [res])

                for run_idx, run in enumerate(runs):
                    if "per_sample_errors" not in run or "per_sample_matrices" not in run:
                        continue

                    errs = np.array(run["per_sample_errors"], dtype=float)
                    mats = run["per_sample_matrices"]

                    all_errors.append(errs)
                    all_mats.append(mats)
                    algo_labels.append(algo)
                    run_metadata.append({'algo': algo, 'run_idx_in_algo': run_idx})

                    if "per_sample_true_labels" in run:
                        all_true_labels.append(run["per_sample_true_labels"])
                    if "per_sample_pred_labels" in run:
                        all_pred_labels.append(run["per_sample_pred_labels"])

            if not all_errors:
                continue

            # Check for label consistency
            labels_available = len(all_true_labels) == len(all_errors)
            if labels_available and len(all_true_labels) > 1:
                first_labels = tuple(all_true_labels[0])
                for i, labels in enumerate(all_true_labels[1:], 1):
                    if tuple(labels) != first_labels:
                        meta0 = run_metadata[0]
                        meta_i = run_metadata[i]
                        raise ValueError(
                            f"Inconsistent 'per_sample_true_labels' in problem '{problem}'. "
                            f"Run {meta0['run_idx_in_algo']} of algorithm '{meta0['algo']}' does not match "
                            f"run {meta_i['run_idx_in_algo']} of algorithm '{meta_i['algo']}'."
                        )

            all_errors = np.stack(all_errors)
            eps = 1e-12
            num_samples = all_errors.shape[1]
            num_runs = all_errors.shape[0]

            # Per-sample min/max across all runs and algorithms
            min_errors = np.nanmin(all_errors, axis=0)
            max_errors = np.nanmax(all_errors, axis=0)
            denom = max_errors - min_errors + eps

            # Best run index per sample (lowest error)
            best_run_idx_per_sample = np.nanargmin(
                np.where(np.isnan(all_errors), np.inf, all_errors), axis=0
            )
            best_mats = [all_mats[idx][j] for j, idx in enumerate(best_run_idx_per_sample)]

            # Extract best predicted labels if available
            best_pred_labels = None
            pred_labels_available = len(all_pred_labels) == len(all_errors)
            if pred_labels_available:
                best_pred_labels = [
                    all_pred_labels[idx][j] for j, idx in enumerate(best_run_idx_per_sample)
                ]

            # Vectorized relative error computation for all runs
            relative_errors = (all_errors - min_errors[None, :]) / denom[None, :]

            # Pre-compute Frobenius distances for all runs at once
            # Convert all matrices to numpy arrays and stack them
            try:
                # Try to stack all matrices if they have the same shape
                best_mats_array = np.array(best_mats, dtype=float)  # Shape: (num_samples, ...)

                # Stack all run matrices: shape (num_runs, num_samples, ...)
                all_mats_stacked = []
                for run_mats in all_mats:
                    run_mats_array = np.array(run_mats[:num_samples], dtype=float)
                    all_mats_stacked.append(run_mats_array)
                all_mats_stacked = np.array(all_mats_stacked)  # (num_runs, num_samples, ...)

                # Compute Frobenius distances vectorized
                # Diff shape: (num_runs, num_samples, ...)
                diff = all_mats_stacked - best_mats_array[None, :, ...]

                # Flatten last dimensions and compute norms
                shape = diff.shape
                diff_flat = diff.reshape(shape[0], shape[1], -1)  # (num_runs, num_samples, flattened)
                best_flat = best_mats_array.reshape(shape[1], -1)  # (num_samples, flattened)

                # Compute norms: (num_runs, num_samples)
                diff_norms = np.linalg.norm(diff_flat, axis=2)
                best_norms = np.linalg.norm(best_flat, axis=1)  # (num_samples,)

                # Frobenius distances: (num_runs, num_samples)
                frobenius_distances = diff_norms / (best_norms[None, :] + eps)

                # Mask out invalid values (NaN errors or infinite Frobenius)
                valid_mask = ~np.isnan(all_errors) & np.isfinite(frobenius_distances)
                frobenius_distances = np.where(valid_mask, frobenius_distances, np.nan)

                matrices_vectorized = True

            except (ValueError, TypeError):
                # Fall back to slower method if matrices have different shapes
                matrices_vectorized = False
                frobenius_distances = np.full((num_runs, num_samples), np.nan)

                for run_idx in range(num_runs):
                    mats = all_mats[run_idx]
                    for j in range(min(num_samples, len(mats))):
                        if np.isnan(all_errors[run_idx, j]):
                            continue

                        try:
                            ma = np.array(mats[j], dtype=float)
                            mb = np.array(best_mats[j], dtype=float)

                            if ma.shape == mb.shape and not (np.any(np.isnan(ma)) or np.any(np.isnan(mb))):
                                frob = np.linalg.norm(ma - mb) / (np.linalg.norm(mb) + eps)
                                if np.isfinite(frob):
                                    frobenius_distances[run_idx, j] = frob
                        except:
                            continue

            # Compute per-algorithm statistics
            summary = {}
            unique_algos = sorted(set(algo_labels))

            for algo in unique_algos:
                # Get indices of all runs for this algorithm
                algo_mask = np.array([a == algo for a in algo_labels])

                # Vectorized relative errors for this algorithm
                algo_rel_errs = relative_errors[algo_mask]  # Shape: (num_algo_runs, num_samples)
                algo_frobs = frobenius_distances[algo_mask]  # Shape: (num_algo_runs, num_samples)

                # Average across samples for each run (ignoring NaNs)
                run_avg_rel_errs = np.nanmean(algo_rel_errs, axis=1)
                run_avg_rel_errs = run_avg_rel_errs[~np.isnan(run_avg_rel_errs)]

                run_avg_frobs = np.nanmean(algo_frobs, axis=1)
                run_avg_frobs = run_avg_frobs[~np.isnan(run_avg_frobs)]

                num_runs = len(run_avg_rel_errs)

                summary[algo] = {
                    "mean_relative_error": float(np.mean(run_avg_rel_errs)) if len(run_avg_rel_errs) > 0 else None,
                    "std_relative_error": float(np.std(run_avg_rel_errs, ddof=1)) if num_runs > 1 else None,
                    "mean_frobenius": float(np.mean(run_avg_frobs)) if len(run_avg_frobs) > 0 else None,
                    "std_frobenius": float(np.std(run_avg_frobs, ddof=1)) if len(run_avg_frobs) > 1 else None,
                    "se_relative_error": float(
                        np.std(run_avg_rel_errs, ddof=1) / np.sqrt(num_runs)) if num_runs > 1 else None,
                    "se_frobenius": float(np.std(run_avg_frobs, ddof=1) / np.sqrt(num_runs)) if len(
                        run_avg_frobs) > 1 else None,
                    "num_runs": num_runs
                }

            problem_summaries[problem] = {
                "num_datapoints": num_samples,
                "num_runs": all_errors.shape[0],
                "algorithms": summary,
                "true_labels": all_true_labels[0] if labels_available else None,
                "best_predicted_labels": best_pred_labels,
            }

        return problem_summaries

    import json
    import os

    results_list = []
    for budget in budgets:
        for algo in all_algos:
            algo_dir = os.path.join(base_results_dir, algo, f"budget_{budget}")
            for method_name in problem_names:
                eval_path = os.path.join(algo_dir, f"eval_{method_name}.json")
                if os.path.exists(eval_path):
                    with open(eval_path, "r") as f:
                        metrics = json.load(f)
                    results_list.append({
                        "Algorithm": algo,
                        "Problem": method_name,
                        "metrics": metrics
                    })

    # Analyze per problem
    analysis = analyze_run_results(results_list)

    # Print summary per problem
    for problem, pdata in analysis.items():
        print(f"\n=== Problem: {problem} ===")
        print(f"Analyzed {pdata['num_datapoints']} datapoints across {pdata['num_runs']} runs")
        for algo, stats in pdata["algorithms"].items():
            print(f"  Algorithm: {algo}")
            print(f"    Mean relative error:   {stats['mean_relative_error']:.6f} ± {stats['std_relative_error']:.6f}")
            print(f"    Mean Frobenius dist.:  {stats['mean_frobenius']:.6f} ± {stats['std_frobenius']:.6f}")

    # %%

    # %%

    # %%

    # %%
    analysis
    # %%
    # create pgd windows and check what causes the nan
    # %%

    # %%
    import numpy as np

    for entry in results_list:
        algo = entry["Algorithm"]
        prob = entry["Problem"]
        metrics = entry["metrics"]

        runs = metrics.get("per_run", [metrics])
        for run_idx, run in enumerate(runs):
            mats = run.get("per_sample_matrices", [])
            for j, m in enumerate(mats):
                m_arr = np.array(m, dtype=float)
                if np.any(np.isnan(m_arr)):
                    print(f"NaN matrix found: Problem={prob}, Algo={algo}, Run={run_idx}, Sample={j}")
                    break
    # %%
    import pandas as pd

    def analysis_to_df(analysis):
        """
        Convert the nested 'analysis' dict from analyze_run_results
        into a long DataFrame suitable for plotting.
        """
        rows = []
        for problem, pdata in analysis.items():
            for algo, stats in pdata["algorithms"].items():
                rows.append({
                    "Problem": problem,
                    "Algorithm": algo,
                    "mean_relative_error": stats.get("mean_relative_error"),
                    "se_relative_error": stats.get("se_relative_error"),
                    "std_relative_error": stats.get("std_relative_error"),
                    "mean_frobenius": stats.get("mean_frobenius"),
                    "se_frobenius": stats.get("se_frobenius"),
                    "std_frobenius": stats.get("std_frobenius"),
                    "num_runs": stats.get("num_runs"),
                    "num_datapoints": pdata.get("num_datapoints")
                })
        df = pd.DataFrame(rows)
        return df

    # %%
    df_analysis = analysis_to_df(analysis)
    df_analysis
    # %%

    # %%

    # %%

    # %%

    # %%
    import matplotlib.pyplot as plt
    import numpy as np

    def plot_df_analysis(df, metric="mean_relative_error", error_col=None):
        """
        Plot results from a flattened analysis DataFrame.

        df: pd.DataFrame with columns ['Problem', 'Algorithm', metric, error_col]
        metric: which mean to plot ('mean_relative_error' or 'mean_frobenius')
        error_col: which column contains the error bars (SEM)
        """
        if error_col is None:
            error_col = "se_relative_error" if metric == "mean_relative_error" else "se_frobenius"

        problem_names = df["Problem"].unique()
        num_problems = len(problem_names)

        fig, axes = plt.subplots(1, num_problems, figsize=(6 * num_problems, 5), squeeze=False)

        for ax, problem in zip(axes[0], problem_names):
            sub = df[df["Problem"] == problem].sort_values("Algorithm")
            algos = sub["Algorithm"].tolist()
            means = sub[metric].tolist()
            errors = sub[error_col].tolist()

            x = np.arange(len(algos))
            ax.bar(x, means, yerr=errors, capsize=5, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(algos, rotation=45, ha="right")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"Problem: {problem}")
            ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.show()

    plot_df_analysis(df_analysis, metric="mean_relative_error")
    # %%
    plot_df_analysis(df_analysis, metric="mean_frobenius")
    # %%
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import os
    import pandas as pd

    def plot_analysis_with_sem(analysis, algorithm_colors, metric="mean_relative_error", W=8, savepath=None):
        """
        Plot per-problem results with horizontal bars and SEM error bars.
        Total figure width is W, height is W/2.2.

        Parameters:
        -----------
        analysis : dict or pd.DataFrame
            Dictionary from analyze_run_results OR a flattened DataFrame with columns:
            ['Problem', 'Algorithm', metric, se_relative_error, se_frobenius']
        algorithm_colors : dict
            Mapping from algorithm name to RGB color tuple
        metric : str
            Which mean to plot ("mean_relative_error" or "mean_frobenius")
        W : float
            Total figure width in inches
        savepath : str
            Path to save the figure (optional)
        """
        # Determine if input is dict or DataFrame
        if isinstance(analysis, dict):
            problem_names = list(analysis.keys())
        elif isinstance(analysis, pd.DataFrame):
            problem_names = analysis["Problem"].unique()
        else:
            raise ValueError("analysis must be a dict or a pandas DataFrame")

        num_problems = len(problem_names)
        if isinstance(W, (int, float)):
            fig_width = W
            fig_height = W / 2.2  # proportional height
        else:
            fig_width, fig_height = W

        fig, axes = plt.subplots(
            1, num_problems,
            figsize=(fig_width, fig_height),
            sharey=False,
            squeeze=False
        )
        axes = axes[0]  # flatten if necessary

        # Metric label
        metric_label = metric.replace("_", " ").title()
        if metric == "mean_relative_error":
            metric_label = "Relative Error"
        elif metric == "mean_frobenius":
            metric_label = "Frobenius Error"

        for idx, ax in enumerate(axes):
            problem = problem_names[idx]

            if isinstance(analysis, dict):
                pdata = analysis[problem]
                algos = sorted(pdata["algorithms"].keys())
                means = []
                sems = []
                for algo in algos:
                    stats = pdata["algorithms"][algo]
                    means.append(stats[metric])
                    sem_val = (
                        stats["se_relative_error"] if metric == "mean_relative_error"
                        else stats["se_frobenius"]
                    )
                    sems.append(sem_val)
            else:  # DataFrame
                sub = analysis[analysis["Problem"] == problem].sort_values("Algorithm")
                algos = sub["Algorithm"].tolist()
                means = sub[metric].tolist()
                sems = sub["se_relative_error"].tolist() if metric == "mean_relative_error" else sub[
                    "se_frobenius"].tolist()

            y = np.arange(len(algos))
            colors = [algorithm_colors.get(a, (0.5, 0.5, 0.5)) for a in algos]

            ax.barh(
                y, means, xerr=sems,
                capsize=1.5,
                color=colors,
                edgecolor="none",
                height=0.7,
                alpha=1,
                error_kw={'elinewidth': 1.5, 'ecolor': 'black', 'capsize': 1}
            )

            ax.set_yticks(y)
            if idx == 0:
                ax.set_yticklabels(algos, fontsize=6)
                ax.set_ylabel("Algorithm", fontsize=10)
            else:
                ax.set_yticklabels([])

            ax.invert_yaxis()
            means_arr = np.array(means)
            sems_arr = np.array([s if s is not None else 0 for s in sems])
            xmin = max(0, (means_arr - sems_arr).min() - 0.03 * means_arr.max())
            xmax = (means_arr + sems_arr).max() + 0.03 * means_arr.max()
            ax.set_xlim(xmin, xmax)

            ax.set_xlabel(metric_label, fontsize=10)
            ax.set_title(f"{problem}", fontsize=11)
            ax.grid(True, axis='x', linestyle='--', alpha=0.5)
            sns.despine(ax=ax, left=True, bottom=False)

        plt.tight_layout(pad=0.5, w_pad=0.2)
        if savepath is not None:
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            plt.savefig(savepath)
        plt.show()

    # --- Usage: pass total width W ---
    path_pdf = os.path.join(figure_path, "mean_relative_error.pdf")
    plot_analysis_with_sem(analysis, algorithm_colors, metric="mean_relative_error", W=W, savepath=path_pdf)

    path_pgf = os.path.join(figure_path, "mean_relative_error.pgf")
    plot_analysis_with_sem(analysis, algorithm_colors, metric="mean_relative_error", W=W, savepath=path_pgf)

    # %%

    # %%

    # %%

    # %%
    path = os.path.join(figure_path, f"mean_frobenius.pdf")
    plot_analysis_with_sem(analysis, algorithm_colors, metric="mean_frobenius", savepath=path, W=W)
    path = os.path.join(figure_path, f"mean_frobenius.pgf")
    plot_analysis_with_sem(analysis, algorithm_colors, metric="mean_frobenius", savepath=path, W=W)

    # %%
    def compute_grouped_scores(df):
        """
        Aggregates per-algorithm statistics across all problems for both metrics.
        """
        # Use agg instead of apply to avoid DeprecationWarning
        grouped = df.groupby("Algorithm").agg(
            mean_relative_error=("mean_relative_error", "mean"),
            se_relative_error=("se_relative_error", lambda x: np.sqrt(
                np.sum((x * np.sqrt(df.loc[x.index, 'num_runs'])) ** 2) / df.loc[x.index, 'num_runs'].sum()) / len(x)),
            mean_frobenius=("mean_frobenius", "mean"),
            se_frobenius=("se_frobenius", lambda x: np.sqrt(
                np.sum((x * np.sqrt(df.loc[x.index, 'num_runs'])) ** 2) / df.loc[x.index, 'num_runs'].sum()) / len(x))
        ).reset_index()
        grouped["Problem"] = "Grouped"
        return grouped

    df_analysis_grouped = compute_grouped_scores(df_analysis)
    # %%
    df_analysis_grouped
    # %%
    # now plot grouped scores with sem
    plot_analysis_with_sem(
        df_analysis_grouped,
        algorithm_colors,
        metric="mean_relative_error",
        savepath=os.path.join(figure_path, "grouped_mean_relative_error.pdf"),
        W=(W / 2, W / 2.2)
    )
    plot_analysis_with_sem(
        df_analysis_grouped,
        algorithm_colors,
        metric="mean_relative_error",
        savepath=os.path.join(figure_path, "grouped_mean_relative_error.pgf"),
        W=(W / 2, W / 2.2)
    )
    # %%
    plot_analysis_with_sem(
        df_analysis_grouped,
        algorithm_colors,
        metric="mean_frobenius",
        savepath=os.path.join(figure_path, "grouped_mean_frobenius.pdf"),
        W=(W / 2, W / 2.2),
    )
    plot_analysis_with_sem(
        df_analysis_grouped,
        algorithm_colors,
        metric="mean_frobenius",
        savepath=os.path.join(figure_path, "grouped_mean_frobenius.pgf"),
        W=(W / 2, W / 2.2),
    )
    # %%
    for budget in budgets:
        print(f"\n=== Best Params Summary | Budget: {budget} ===")
        for algo in all_algos:
            algo_dir = os.path.join(base_results_dir, algo, f"budget_{budget}")
            param_path = os.path.join(algo_dir, "best.yml")

            if os.path.exists(param_path):
                stored_params = load_params(param_path)
                print(f"\n--- {algo} ---")
                for k, v in stored_params.items():
                    print(f"{k}: {v}")
            else:
                print(f"\n--- {algo} ---")
                print("No stored params found.")

    # %%

    def _cost_shgo(n_init: int, local_runs: int, local_max_steps: int, grad_weight: int) -> int:
        if local_max_steps == 0:
            return n_init
        return n_init + local_runs * local_max_steps * grad_weight + local_runs

    def _cost_parallel_sa(par_runs: int, max_iter: int) -> int:
        return par_runs * (max_iter + 1)

    def _cost_es(pop: int, iters: int) -> int:
        return pop * (iters + 1)

    def _cost_pso(swarm: int, steps: int) -> int:
        return swarm * (steps + 1)

    def _cost_cd(dim: int, samples: int) -> int:
        return dim * samples

    def _cost_wcd(dim: int, base: int, first_factor: int, rounds: int) -> int:
        per_round = base * (dim - 1 + first_factor)
        return per_round * rounds

    # NEW cost helper for parallel gradient methods
    def _cost_pgd(par_runs: int, max_iter: int, grad_weight: int) -> int:
        return par_runs * max_iter * grad_weight + par_runs

    # --- ITS cost helper (already defined) ---
    def _cost_its(n_samples: int, n_hypotheses: int, dim: int) -> int:
        return n_samples * (1 + n_hypotheses * (dim - 1))

    # %%

    # %%

    # %%

    # %%

    # %%

    # %% md

    # %%
    if dataset == "bigger_mnist":
        from search.parallel_gradient import ParallelGradientDescent

        pgd = ParallelGradientDescent(max_iterations=5, parallel_runs=6, )
        transform_seq.transformations

        problem_knn_new = replace_rotation_transforms(problem_nn_pytorch_pretrained)
        print()
        res1 = load_or_run_evaluate_confidence_and_search(
            model=model,
            optimizer=pgd,
            problem=problem_nn_pytorch_pretrained,
            test_loader=test_loader_transformed,
            save_path=os.path.join(current_path, "experiment_files", "export", "results", "comparision_search_datasets",
                                   dataset, transform_name, "pgd_bigger_mnist_knn_per_class.json"),
            # auto-loads if exists; saves after run otherwise
            max_batch_override=dataset_info
            .batch_size_search,
            show_progress=True,
            repeats=4,
            return_per_run=True,
            overwrite=False, store_val=False
        )
        res2 = load_or_run_evaluate_confidence_and_search(
            model=model,
            optimizer=pgd,
            problem=problem_knn_new,
            test_loader=test_loader_transformed,
            save_path=os.path.join(current_path, "experiment_files", "export", "results", "comparision_search_datasets",
                                   dataset, transform_name, "pgd_bigger_mnist_knn_per_class_new_transforms.json"),  #

            max_batch_override=dataset_info
            .batch_size_search,
            show_progress=True,
            repeats=4,
            return_per_run=True,
            overwrite=False, store_val=False
        )
        # plot results
        print(res1)
        print(res2)
        plt.figure()
        # barplot
        labels = ['Original Transforms', 'New Transforms']
        means = [res1['accuracy_mean'], res2['accuracy_mean']]
        se = [res1['accuracy_se'], res2['accuracy_se']]
        x = np.arange(len(labels))
        plt.bar(x, means, yerr=se, capsize=5, alpha=0.8)
        plt.xticks(x, labels)
        plt.ylabel('Accuracy')
        plt.title('Comparison of PGD with KNN Confidence')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    # %%

    # %%
    if dataset == "modelnet10":
        from utils.affine_transforms import AffineTransformations3D

        pgd = ParallelGradientDescent(max_iterations=5, parallel_runs=6, )

        problem_knn_new = replace_rotation_transforms(problem_nn_pytorch_pretrained)

        print()
        res1 = load_or_run_evaluate_confidence_and_search(
            model=model,
            optimizer=pgd,
            problem=problem_nn_pytorch_pretrained,
            test_loader=test_loader_transformed,
            save_path=os.path.join(current_path, "experiment_files", "export", "results", "comparision_search_datasets",
                                   dataset, transform_name, "pgd_bigger_mnist_knn_per_class.json"),
            # auto-loads if exists; saves after run otherwise
            max_batch_override=dataset_info
            .batch_size_search,
            show_progress=True,
            repeats=4,
            return_per_run=True,
            overwrite=False, store_val=False
        )
        print(res1)
        print()
        res2 = load_or_run_evaluate_confidence_and_search(
            model=model,
            optimizer=pgd,
            problem=problem_knn_new,
            test_loader=test_loader_transformed,
            save_path=os.path.join(current_path, "experiment_files", "export", "results", "comparision_search_datasets",
                                   dataset, transform_name, "pgd_bigger_mnist_knn_per_class_new_transforms.json"),  #

            max_batch_override=dataset_info
            .batch_size_search,
            show_progress=True,
            repeats=4,
            return_per_run=True,
            overwrite=False, store_val=False
        )
        print(res2)
        print()
        pgd = ParallelGradientDescent(max_iterations=5, parallel_runs=6, learning_rate=0.1)

        problem_knn_3 = replace_rotation_transforms_2vec(problem_nn_pytorch_pretrained)
        res3 = load_or_run_evaluate_confidence_and_search(
            model=model,
            optimizer=pgd,
            problem=problem_knn_3,
            test_loader=test_loader_transformed,
            save_path=os.path.join(current_path, "experiment_files", "export", "results", "comparision_search_datasets",
                                   dataset, transform_name, "pgd_bigger_mnist_knn_per_class_new_transforms_3dvec.json"),
            max_batch_override=dataset_info
            .batch_size_search,
            show_progress=True,
            repeats=4,
            return_per_run=True,
            overwrite=False, store_val=False
        )
        print(res3)
    # %%
    # disable p grad
    for param in model.parameters():
        param.requires_grad = False

    # %%
    class EvalCounterWrapper(nn.Module):
        def __init__(self, model):
            super(EvalCounterWrapper, self).__init__()
            self.model = model
            self.eval_count = 0

        def __call__(self, *args, **kwargs):
            batch = args[0] if args else kwargs.get("x", None)
            batch_size = len(batch) if batch is not None else 1

            # Count once if in no_grad, twice otherwise
            if torch.is_grad_enabled():
                self.eval_count += 2 * batch_size
            else:
                self.eval_count += batch_size

            return self.model(*args, **kwargs)

        def reset(self):
            self.eval_count = 0

    # %%
    # compare model with forward and with forward and backward to inputs
    # disble p grad
    import time
    for param in model.parameters():
        param.requires_grad = False

    x = next(iter(test_loader_transformed))[0].to(device)

    # first forward
    start = time.time()
    for i in range(100):
        with torch.no_grad():
            _ = model(x)
            torch.cuda.synchronize() if device.type == 'cuda' else None
        end = time.time()
    # then forward and backward
    res1 = end - start
    start = time.time()
    for i in range(100):
        x.requires_grad = True

        out = model(x)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize() if device.type == 'cuda' else None
    end = time.time()
    res2 = end - start

    print(f"Forward only time per pass: {res1 / 100:.6f}s")
    print(f"Forward + Backward time per pass: {res2 / 100:.6f}s")
    print(f"Ratio (F+B) / F: {res2 / res1:.2f}")

    # %%

    # %%
    import time
    import os
    import torch
    import torch.nn as nn

    # =========================
    # CONFIG
    # =========================
    measure_time = True  # set False to skip the timing benchmark
    num_batches = 5  # measured batches (excluding first, plus one extra at end)

    # Warm-up the model once to initialize kernels
    batch_warmup = next(iter(test_loader_transformed))[0].to(device)
    with torch.no_grad():
        _ = model(batch_warmup)
    print("Model warm-up done.\n")

    # =========================
    # BENCHMARK 1: Eval count / cost check
    # =========================
    print("=== Eval Count / Budget Checking ===")
    for budget in budgets:
        print(f"\n--- Budget: {budget} ---")
        for algo in all_algos:
            try:
                if algo == "its" or algo in ["shgo_individual", "wcd_lat_ind"]:
                    continue
                if algo in complex_algos:
                    continue

                # Load params
                algo_dir = os.path.join(base_results_dir, algo, f"budget_{budget}")
                param_path = os.path.join(algo_dir, "best.yml")
                if os.path.exists(param_path):
                    stored_params = load_params(param_path)
                else:
                    stored_params = get_default_params(algo, budget)

                # Compute cost
                if algo == "shgo":
                    cost = _cost_shgo(
                        stored_params["shgo_initial_samples"],
                        stored_params["shgo_local_runs"],
                        stored_params["shgo_local_steps"],
                        stored_params.get("grad_weight", 2),
                    )
                elif algo == "parallel_sa":
                    cost = _cost_parallel_sa(
                        stored_params["psa_parallel_runs"],
                        stored_params["psa_max_iterations"],
                    )
                elif algo == "evolutionary":
                    cost = _cost_es(
                        stored_params["es_population"],
                        stored_params["es_iters"],
                    )
                elif algo == "pso":
                    cost = _cost_pso(
                        stored_params["pso_swarm_size"],
                        stored_params["pso_steps"],
                    )
                elif algo == "cd":
                    cost = None
                elif algo == "wcd":
                    cost = None
                elif algo in {"pgd", "pgd_restart", "pgd_window"}:
                    cost = _cost_pgd(
                        stored_params["pgd_parallel_runs"],
                        stored_params["pgd_max_iterations"],
                        stored_params.get("grad_weight", 2),
                    )
                elif algo == "random_search":
                    cost = budget
                else:
                    cost = None

                # Build a fresh SinglePass energy problem to count model evals
                model_counter = EvalCounterWrapper(model)
                energy_counter_confidence = SinglePassConfidence(model_counter, EnergyConfidence(), index=None)
                problem_energy_counter = TransformationProblem(
                    energy_counter_confidence, transform_seq, consolidate_method="consolidate_simple"
                )
                problem_energy_counter.max_batch_size = dataset_info.batch_size_search

                # Run a single batch to count evaluations
                batch = next(iter(test_loader_transformed))
                model_counter.reset()
                search_obj = build_search_algorithm(algo, stored_params, problem=problems[0], budget=budget,
                                                    model=model_counter)
                search_obj.optimize(problem_energy_counter, x=batch[0].to(device))

                print(f"Algorithm: {algo}")
                print(f"  Computed cost: {cost}")
                print(f"  Budget: {budget}")
                print(f"  Eval count per sample: {model_counter.eval_count / len(batch[0]):.2f}")
            except Exception as e:
                print(f"Error processing algorithm {algo} at budget {budget}: {e}")

    # %%
    search_obj.optimize(problem_energy_counter, x=batch[0].to(device))
    # %%
    search_obj.num_iters
    # %%
    from search.objective_generators import _copy_problem_with_init_method
    # First code block (Original)

    import time
    import os
    import torch

    # Number of batches for benchmarking
    num_batches = 5  # actual measured batches per algorithm will be num_batches + 1 (warmup/excluded)

    # Warm-up model: run a single forward pass to initialize CUDA kernels, memory allocations, etc.
    batch_warmup = next(iter(test_loader_transformed))[0].to(device)
    with torch.no_grad():
        _ = model(batch_warmup)

    print("Warm-up done.\n")

    # Store benchmark results
    benchmark_results = {}

    for budget in budgets:
        print(f"\n=== Budget: {budget} ===")
        for algo in all_algos:
            print(f"\n--- Algorithm: {algo} ---")

            # Determine problems to use
            if algo in complex_algos:
                current_problems = problems_complex
                current_problem_names = problem_names_complex
            elif algo in ["shgo_individual", "wcd_lat_ind"]:
                current_problems = problems_individual
                current_problem_names = problem_names_individual
            else:
                current_problems = problems
                current_problem_names = problem_names

            # Map algorithm names for paths
            if algo in complex_algos:
                algo_name_for_path = algo_variant_mapping[algo]
            elif algo == "shgo_individual":
                algo_name_for_path = "shgo"
            elif algo == "wcd_lat_ind":
                algo_name_for_path = "wcd_lattice"
            else:
                algo_name_for_path = algo

            # Load or set parameters
            algo_dir = os.path.join(base_results_dir, algo, f"budget_{budget}")
            os.makedirs(algo_dir, exist_ok=True)
            param_path = os.path.join(algo_dir, "best.yml")
            stored_params = load_params(param_path) if os.path.exists(param_path) else get_default_params(
                algo_name_for_path, budget)

            search_obj = build_search_algorithm(
                algo_name_for_path,
                stored_params,
                problem=current_problems[0],  # any problem for construction
                budget=budget,
                model=model,
            )

            for prob, prob_name in zip(current_problems, current_problem_names):
                prob.max_batch_size = dataset_info.batch_size_search

                if algo == "wcd_lat_ind":
                    print("Setting lattice basis for WCD Lattice Individual")
                    prob = _copy_problem_with_init_method(
                        prob,
                        init_method="permuted_lattice"
                    )

                batch_times = []
                batch_eval_counts = []

                data_iter = iter(test_loader_transformed)

                # Skip first batch (warmup per algorithm)
                x_first = next(data_iter)

                # Run timed batches
                for _ in range(num_batches):
                    batch = next(data_iter)
                    x_batch = batch[0].to(device)

                    start_time = time.time()
                    search_obj.optimize(prob, x_batch)
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    end_time = time.time()

                    batch_times.append(end_time - start_time)

                # Add one extra batch at the end
                batch = next(data_iter)
                x_batch = batch[0].to(device)
                start_time = time.time()
                search_obj.optimize(prob, x_batch)
                end_time = time.time()
                batch_times.append(end_time - start_time)

                avg_time = sum(batch_times) / len(batch_times)

                benchmark_results.setdefault(algo, {})[prob_name] = {
                    "avg_time_per_batch": avg_time,
                }

                print(f"[{prob_name}] Avg time: {avg_time:.4f}s")

    # %%
    print("\n=== Benchmark Summary ===")
    for algo, probs in benchmark_results.items():
        print(f"\nAlgorithm: {algo}")
        for prob_name, metrics in probs.items():
            print(f"  {prob_name}: Time {metrics['avg_time_per_batch']:.4f}s")
    # %%

    # %%

    # %%
    import json
    import os
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from collections import defaultdict

    # === Locate base path ===
    path_found = False
    current_path = os.getcwd()
    while not path_found:
        if os.path.exists(os.path.join(current_path, "experiment_files")):
            path_found = True
            break
        current_path = os.path.dirname(current_path)

    # === Import to get transform names dynamically ===
    from experiment_thesis.dataset_preperation.get_dataset import get_dataset_info

    # === Configuration ===
    datasets = ["bigger_mnist", "coil100", "bigger_emnist", "modelnet10"]
    budget = 60
    base_path = Path(current_path) / "experiment_files" / "search_results"

    # === Architecture mapping ===
    default_architecture_mapping = {
        "mnist": "resnet_small",
        "bigger_mnist": "resnet_small",
        "emnist": "extended_resnet_small",
        "bigger_emnist": "bigger_extended_resnet_small",
        "coil100": "coil_resnet_small",
        "tu_berlin": "bi_lstm",
        "modelnet10": "pointnetplus",
    }

    problem_names = ["logit_energy", "knn_per_class", "learned_energy"]

    # === Collect all data ===
    # Structure: data_dict[algo][problem][dataset] = list of (accuracy, se)
    data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for dataset in datasets:
        dataset_info = get_dataset_info(dataset)
        transform_name = dataset_info.transform_seq_name
        architecture = default_architecture_mapping[dataset]

        dataset_path = base_path / dataset / transform_name / architecture
        print(f"Checking path for {dataset}: {dataset_path}")

        if not dataset_path.exists():
            print(f"Warning: Path does not exist for {dataset}: {dataset_path}")
            continue

        for algo_dir in dataset_path.iterdir():
            if not algo_dir.is_dir():
                continue

            algo_name = algo_dir.name
            budget_dir = algo_dir / f"budget_{budget}"

            if not budget_dir.exists():
                continue

            for problem_name in problem_names:
                eval_file = budget_dir / f"eval_{problem_name}.json"
                if not eval_file.exists():
                    continue

                try:
                    with open(eval_file, 'r') as f:
                        metrics = json.load(f)

                    accuracy = metrics.get("accuracy_mean")
                    accuracy_se = metrics.get("accuracy_se")

                    if accuracy is not None:
                        data_dict[algo_name][problem_name][dataset].append(
                            {"mean": accuracy, "se": accuracy_se}
                        )

                except Exception as e:
                    print(f"Error reading {eval_file}: {e}")
    # %%
    # === Identify algorithms that appear in ALL datasets ===
    all_algorithms = set(data_dict.keys())
    algorithms_in_all = set()

    for algo in all_algorithms:
        datasets_with_algo = set()
        for problem in problem_names:
            datasets_with_algo.update(data_dict[algo][problem].keys())

        if datasets_with_algo == set(datasets):
            algorithms_in_all.add(algo)

    print(f"\nAlgorithms appearing in all datasets: {sorted(algorithms_in_all)}")

    # === Compute per-dataset statistics ===
    results = []

    for algo in sorted(algorithms_in_all):
        for problem in problem_names:
            for dataset in datasets:
                entries = data_dict[algo][problem][dataset]
                if not entries:
                    continue

                # Extract values
                accuracies = [e["mean"] for e in entries if e["mean"] is not None]
                ses = [e["se"] for e in entries if e["se"] is not None]

                if accuracies:
                    mean_acc = np.mean(accuracies)
                    std_acc = np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0
                    se_acc = np.mean(ses) if ses else std_acc / np.sqrt(len(accuracies))

                    results.append({
                        "Dataset": dataset,
                        "Algorithm": algo,
                        "Problem": problem,
                        "Mean_Accuracy": mean_acc,
                        "Std_Accuracy": std_acc,
                        "SE_Accuracy": se_acc,
                        "N_Samples": len(accuracies),
                    })

    # === Create and display DataFrame ===
    df_results = pd.DataFrame(results)

    if not df_results.empty:
        print("\n=== Results Per Dataset ===\n")
        for dataset in datasets:
            print(f"\n=== Dataset: {dataset} ===")
            for problem in problem_names:
                subset = df_results[
                    (df_results["Dataset"] == dataset) &
                    (df_results["Problem"] == problem)
                    ].sort_values("Mean_Accuracy", ascending=False)
                if not subset.empty:
                    print(f"\n--- Problem: {problem} ---")
                    print(subset.to_string(index=False))

        # Save to CSV
        output_file = base_path / "aggregated_results_per_dataset.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    else:
        print("\nNo results found. Check if the paths are correct and data exists.")


def run_2_2():
    # %%
    import copy

    import torch
    import torch.nn as nn
    import torchvision
    import numpy as np
    from matplotlib import pyplot as plt

    from its.search import InverseTransformationSearch
    from search.parallel_gradient import ParallelGradientDescent
    from utils.affine_transforms_old import AffineTransformation2D
    from utils.sampling import BatchNegativeSampler

    # torch.cuda.is_available = lambda: False
    # device = torch.device("cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # look for experiment files in parents
    import os

    path_found = False
    current_path = os.getcwd()
    while not path_found:
        if os.path.exists(os.path.join(current_path, "experiment_files")):
            path_found = True
            break
        current_path = os.path.dirname(current_path)

    experiment_files_path_data = os.path.join(current_path, "experiment_files", "data")
    dataset = "modelnet10"

    default_architecutre_mapping = {
        "mnist": "resnet_small",
        "bigger_mnist": "resnet_small",
        "emnist": "extended_resnet_small",
        "bigger_emnist": "bigger_extended_resnet_small",
        "coil100": "coil_resnet_small",
        "tu_berlin": "bi_lstm",
        "modelnet10": "pointnetplus",
    }

    architecture = default_architecutre_mapping[dataset]
    budget = None
    # %%
    from utils.transforms.apply import grid_resample_border, grid_resample_reflection
    # %%

    # %%
    from experiment_thesis.dataset_preperation.get_dataset import get_dataset_info, get_dataset

    dataset_info = get_dataset_info(dataset)

    dataset_dict = get_dataset(dataset_info, path=experiment_files_path_data, batch_size=dataset_info.batch_size)
    transform_name = dataset_info.transform_seq_name
    # %%

    # %%

    # %%

    # %%

    # %%

    dataset_dict.keys()
    dataset_train = dataset_dict['train_dataset']
    dataset_val = dataset_dict['val_dataset']
    dataset_test = dataset_dict['test_dataset']
    train_loader = dataset_dict['train_loader']
    val_loader = dataset_dict['val_loader']
    test_loader = dataset_dict['test_loader']
    n_classes = dataset_info.num_classes
    train_loader_transformed = dataset_dict['train_loader_transformed']
    val_loader_transformed = dataset_dict['val_loader_transformed']
    test_loader_transformed = dataset_dict['test_loader_transformed']
    train_loader_no_shuffle = dataset_dict['train_loader_no_shuffle']
    # %%
    x = next(iter(test_loader_transformed))[0]
    # %%

    # %%
    batch_size = next(iter(train_loader))[0].shape[0]

    # %%
    from utils.eval.vis import vis_dataset

    vis_dataset(train_loader, val_loader, test_loader_transformed)
    # %%
    from experiment_thesis.main import train_and_get_model, train_or_load_energy_model
    from experiment_thesis.dataset_preperation.basic_networks import get_network
    from utils.eval.main_model import evaluate_base_model

    model_dir_path = os.path.join(current_path, "experiment_files", "models")
    embedding_cache_path = os.path.join(current_path, "experiment_files", "embedding_cache")
    # Add results dir and helper for save paths
    results_dir_path = os.path.join(current_path, "experiment_files", "results", dataset, architecture,
                                    "comparision_over_budget")
    os.makedirs(results_dir_path, exist_ok=True)

    def savepath(label: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
        return os.path.join(results_dir_path, transform_name, f"{safe}.json")

    # %%
    model = get_network(dataset_info, architecture, num_classes=n_classes).to(device)
    modelname = f"{dataset}_{architecture}"
    cache_name_train = f"{dataset}_{architecture}_embedding_cache_train"

    train_and_get_model(model, model_dir_path, modelname, train_loader, val_loader, trainer_kwargs={
        "accelerator": "auto",
        "max_epochs": dataset_info.epochs,
        "precision": "16-mixed",
    }, load_if_exists=True)

    # %%
    model.eval().to(device)
    # %%
    res = evaluate_base_model(model, test_loader_transformed, device)
    print(res)
    # %%
    res = evaluate_base_model(model, test_loader, device)
    print(res)

    # %%
    class TensorGeometricModelUnwrapper(torch.nn.Module):
        """
        Wrapper for a torch_geometric model that receives a tuple of (pos, y) as input and creates
        a Data object from it that is passed to the model.
        """

        def __init__(self):
            super(TensorGeometricModelUnwrapper, self).__init__()

        def forward(self, data):
            # pos and y are batched tensors from a DataLoader
            # need to reconstruct the original Data objects for torch_geometric models

            pos = data.pos
            batch = data.batch
            # split pos into individual tensors based on batch
            pos_list = torch.split(pos, torch.bincount(batch).tolist())
            return torch.stack(pos_list)

    # %%
    # chek if data is iamge data
    is_image_data = len(dataset_info.input_size) == 3 and dataset_info.input_size[0] in [1, 3]
    # %%
    from utils.transforms.apply import grid_resample
    from experiment_thesis.dataset_preperation.transformation import get_transformation_sequence_images

    transform_seq = get_transformation_sequence_images(
        name=dataset_info.transform_seq_name,
        resample_method=dataset_info.resample_method,
        init_method="sobol"
    ).to(device)
    # %%
    from experiment_thesis.dataset_preperation.basic_networks import get_network_layer

    layer, layer_io = get_network_layer(dataset_info, architecture, 0, num_classes=None, num_rotations=8)
    # %%
    from confidence.direct.logit_based import EnergyConfidence
    from utils.transformation_problem import TransformationProblem
    from confidence.model.single_pass import SinglePassConfidence

    logit_energy = SinglePassConfidence(model, EnergyConfidence(), index=None)
    problem_energy_logits = TransformationProblem(logit_energy, transform_seq, consolidate_method="consolidate_simple")
    # test ot
    from search.shgo import SHGO
    random_search = SHGO(initial_samples=120, local_max_steps=0)

    from utils.eval.ood_performance import load_or_run_evaluate_confidence_and_search, evaluate_confidence_and_search, \
        ITSWRAPPER

    # %%
    load_or_run_evaluate_confidence_and_search(
        model, optimizer=random_search, problem=problem_energy_logits,
        test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
        save_path=savepath("energy_confidence_transformed"), show_progress=True,
        repeats=1)
    # %%
    model.to(device).eval()
    # %%
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # %%

    # %%

    # %%

    # %%
    if "modelnet10" in dataset:
        for init_method in ["individual", "sobol"]:
            print(f"Testing init method: {init_method}")
            import torch
            import kornia
            import matplotlib.pyplot as plt
            import numpy as np
            from mpl_toolkits.mplot3d import Axes3D
            from utils.affine_transforms import AffineTransformations3D
            from utils.transform_sequence import TransformSequence
            from scipy import stats

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Save original init method and temporarily switch to "individual"
            original_init = transform_seq.init_method
            transform_seq.init_method = init_method

            # --- 1. Generate your matrices ---
            zw = transform_seq.initial_param(1280)
            transform_seq.init_method = original_init
            matrix = transform_seq(zw)

            # --- 2. Extract rotation submatrices ---
            R = matrix[:, :3, :3]  # [N, 3, 3]

            # --- 3. Convert to axis–angle representation ---
            rotvec = kornia.geometry.conversions.rotation_matrix_to_axis_angle(R)  # [N, 3]
            theta = torch.linalg.norm(rotvec, dim=1)
            axis = torch.nn.functional.normalize(rotvec, dim=1)

            axis_np = axis.cpu().numpy()
            theta_np = theta.cpu().numpy()

            # --- 4. Quantitative checks ---

            # (a) Axis uniformity
            mean_axis = axis_np.mean(axis=0)
            spherical_var = 1 - np.linalg.norm(mean_axis)
            print("Axis mean vector:", mean_axis)
            print("Spherical variance (1 - |mean|):", spherical_var)

            # (b) Rotation angle
            sample_mean = np.mean(theta_np)
            theory_mean = np.pi / 2 + 2 / np.pi  # Correct expected mean for uniform SO(3)
            print(f"Sample mean rotation angle (rad): {sample_mean:.4f}")
            print(f"Theoretical mean rotation angle (rad): {theory_mean:.4f}")

            # (c) Kolmogorov–Smirnov test against theoretical CDF
            def F_theta(t):
                # CDF for p(theta) ∝ sin²(theta/2)
                return (t - np.sin(t)) / np.pi

            ks_stat, pval = stats.kstest(theta_np, F_theta)
            print(f"KS statistic: {ks_stat:.4f}, p-value: {pval:.4f}")

            # --- 5. Visualization ---

            fig = plt.figure(figsize=(12, 5))

            # Angle histogram
            plt.subplot(1, 2, 1)
            plt.hist(theta_np, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black')
            t_grid = np.linspace(0, np.pi, 200)
            plt.plot(t_grid, (2 / np.pi) * np.sin(t_grid / 2) ** 2, 'r-', lw=2, label='theoretical pdf')
            plt.xlabel("Rotation angle (radians)")
            plt.ylabel("Density")
            plt.title("Rotation Angle Distribution")
            plt.legend()

            # 3D scatter of axes
            ax = fig.add_subplot(1, 2, 2, projection='3d')
            ax.scatter(axis_np[:, 0], axis_np[:, 1], axis_np[:, 2], s=8, alpha=0.7)
            # Draw reference sphere
            u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
            x = np.cos(u) * np.sin(v)
            y = np.sin(u) * np.sin(v)
            z = np.cos(v)
            ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)
            ax.set_box_aspect([1, 1, 1])
            ax.set_title("Rotation Axes on Unit Sphere")

            plt.tight_layout()
            plt.show()

    # %%

    # %%
    from utils.augments import ComposeAugmentations, random_gaussian_noise, random_contrast, \
        random_gamma, random_blur_or_sharpen, build_default_augmentations
    import utils.augments

    def dec_strat(x, idd, y_true):
        out = model(x)
        eq = out.argmax(dim=-1) == y_true
        # convert to tensor where y>=0 if correct, y<0 if incorrect
        y = torch.where(eq, y_true, -1)
        return y

    from utils.augments import build_default_augmentations, small_affine_augment_2d
    from utils.sampling_strategy import GaussianSamplingStrategyLatent, TransformLatentSamplingStrategy
    import importlib
    import utils.sampling_strategy
    import utils.sampling

    importlib.reload(utils.sampling)
    from utils.sampling import BatchNegativeSampler

    energy_model2 = get_network(dataset_info, architecture, num_classes=1).to(device)

    from experiment_thesis.main import train_or_load_energy_model

    if is_image_data:
        transform_true_function = small_affine_augment_2d
        affine_augment = utils.augments.build_default_augmentations()
    else:
        transform_true_function = None
        affine_augment = None

    negative_sampling_module = BatchNegativeSampler(
        TransformLatentSamplingStrategy(
            transform_sequence=transform_seq, ), transform_true_function
        =transform_true_function, augment_function=affine_augment,
        decision_strategy=dec_strat,
    )

    energy_conf2 = train_or_load_energy_model(
        energy_model2, model_dir_path, f"{modelname}_energy2", train_loader,
        val_loader, trainer_kwargs={
            "accelerator": "auto",
            "max_epochs": dataset_info.epochs // 2,
            "precision": "16-mixed" if dataset_info.name not in ["modelnet10"] else "32",
        }, negative_sampling_module=negative_sampling_module, load_if_exists=True)

    # %%
    model.to(device).eval()
    # %%
    from model.pointnet_plus import SAModule
    def set_deterministic_fps(model, random_start=False):
        for module in model.modules():
            if isinstance(module, SAModule):
                module.random_start = random_start
                print(f"Set random_start={random_start} for {module.__class__.__name__}")

    set_deterministic_fps(model)
    set_deterministic_fps(energy_model2)
    # %%
    energy_conf2.to(device).eval()

    problem_energy2 = TransformationProblem(energy_conf2, transform_seq, consolidate_method="consolidate_simple")

    # %%
    load_or_run_evaluate_confidence_and_search(
        model, optimizer=random_search, problem=problem_energy2,
        test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
        save_path=savepath("learned_energy_confidence_transformed"), show_progress=True,
        repeats=1)
    # %%
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # %%

    # %%
    from torch.utils.data import SequentialSampler
    from embedding_cache import LayerEmbeddingCache

    cache_name_train = f"{dataset}_{architecture}_{transform_name}_embedding_cache_train"

    cache_train = LayerEmbeddingCache(model, train_loader_no_shuffle,
                                      cache_dir=os.path.join(embedding_cache_path, cache_name_train))

    dual_output_model = cache_train.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True)
    embeddings_t, final_t, classes_t = cache_train.__call__(layer, capture_modes=layer_io, flatten=True)

    from utils.transformation_problem import TransformationProblem
    from confidence.model.single_pass import SinglePassConfidence
    from confidence.direct.logit_based import EnergyConfidence
    from confidence.control.split import SplitConfidence, PredictedSplitConfidence
    from confidence.unsupervised.classic.nn_pytorch import KNNConfidence, PerClassKNNConfidence

    from confidence.input_transform import InputTransformImage, PCAInputModule, RandomProjectionModule

    nn_pytorch_pretrained = PerClassKNNConfidence(metric="cosine", input_transform=None, computation_mode="masked", k=1)
    nn_pytorch_pretrained.fit(embeddings_t, classes_t)
    nn_pytorch_pretrained.to(device)

    conf_split_pretrained = PredictedSplitConfidence(nn_pytorch_pretrained, EnergyConfidence(), mult=False, b=0.0)
    conf_mod_nn_pytorch_pretrained = SinglePassConfidence(dual_output_model, conf_split_pretrained, index=1)
    problem_nn_pytorch_pretrained = TransformationProblem(conf_mod_nn_pytorch_pretrained, transform_seq,
                                                          consolidate_method="consolidate_simple")
    model.eval().to(device)
    # %%

    # %%
    # benchmark model and dual output model
    # %%
    load_or_run_evaluate_confidence_and_search(
        model, optimizer=random_search, problem=problem_nn_pytorch_pretrained,
        test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
        save_path=savepath("knn_per_class_confidence_transformed"), show_progress=True,
        repeats=1, overwrite=False)
    # %%

    # %%

    # %%

    # %%
    gc.collect()
    torch.cuda.empty_cache()
    # %%
    x = next(iter(train_loader_no_shuffle))

    res1 = dual_output_model(x[0].to(device))[0].cpu().detach().numpy()

    res2 = embeddings_t[:x[0].shape[0]].cpu().detach().numpy()
    if not np.allclose(res1, res2, rtol=1e-5, atol=1e-5):
        raise ValueError("Model is not deterministic!")
    else:
        print("Model is deterministic.")

    del res1
    del res2
    del x
    # %%
    from utils.eval.ood_performance import ITSWRAPPER
    import importlib
    import its.search
    importlib.reload(its.search)

    its2 = ITSWRAPPER(its.search.InverseTransformationSearch(model, None, None, n_hypotheses=1, n_samples=10, extend=0,
                                                             gaussian_filter_channel_wise=True))
    # %%
    from search.tree import CoordinateDescent
    cd = CoordinateDescent()
    # %%
    gc.collect(

    )
    torch.cuda.empty_cache()
    # %%

    # %%

    # %%
    gc.collect()
    torch.cuda.empty_cache()
    # %%
    x = next(iter(test_loader_transformed))[0].to(device)
    # %%

    # %%
    from utils.affine_transforms import AffineTransformations3D, AffineTransformation2D
    from utils.transform_sequence import TransformSequence
    from utils.transformation_problem import TransformationProblem

    # %%

    # %%
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # %%

    # %%

    # %%
    from utils.replacer import replace_rotation_transforms, replace_rotation_transforms_2vec
    import os
    import optuna
    import torch
    import gc
    import numpy as np
    from torch.utils.data import Subset, DataLoader

    from search.objective_generators import (
        make_search_objective,
        save_best_trial_params,
        build_search_algorithm,
        _cost_shgo,  # cost helper for sanity checks
    )
    from search.config import load_params, get_default_params, save_params
    from utils.eval.ood_performance import load_or_run_evaluate_confidence_and_search

    model.eval().to(device)
    # detach model for efficent gradients
    for param in model.parameters():
        param.requires_grad = False

    for param in energy_model2.parameters():
        param.requires_grad = False

    # Algorithms and budgets
    all_algos = ["shgo_c", "wcd_lattice", "its", "random_search"]
    # "cmaes" #cmaes seems currently alway to default to only using 0 iterations.
    # add "its" if you have an ITS optimizer/module wired up

    # for bigger mnist and bigger emnist add shgo
    if dataset in ["bigger_mnist", "bigger_emnist", "coil100"]:
        all_algos.append("shgo")

    # remove pgd pgd restarts and pgd window size for tu berlin
    if dataset in ["tu_berlin"]:
        all_algos = ["wcd_lattice", "its", "random_search"]
        grad_weight = 99999999
    elif dataset in ["modelnet10"]:
        all_algos = ["shgo_vector", "wcd_lat_ind", "random_search", "its", "shgo", "shgo_c"]
        grad_weight = 2
    else:
        grad_weight = 2

    # Add complex/quaternion variants
    complex_algos = ["shgo_c", "pgd_c", "shgo_c_3"]

    budgets = [8, 15, 30, 60, 120, 240]

    grad_weight_algos = {"shgo", "shgo_individual", "pgd", "pgd_restart", "pgd_window",
                         "shgo_c", "pgd_c", "pgd_restart_c", "shgo_c_3", "pgd_c_3", "pgd_vector", "shgo_vector"}
    default_trials = 30
    eval_repeats = 4
    if dataset == "modelnet10" or dataset == "tu_berlin" or dataset == "coil100":
        eval_repeats = 8
    show_progress = True

    # Problem configurations
    problems = [problem_energy_logits, problem_nn_pytorch_pretrained, problem_energy2]
    problem_names = ["logit_energy", "knn_per_class", "learned_energy"]

    # Create complex/quaternion versions of problems
    problems_complex = [replace_rotation_transforms(p) for p in problems]
    problem_names_complex = problem_names

    # New problems for shgo_individual
    problems_individual = [ITSWRAPPER._prepare_problem(p) for p in problems]
    problem_names_individual = problem_names

    problems_rotvec = [replace_rotation_transforms_2vec(p) for p in problems]
    problem_names_rotvec = problem_names

    for p in problems:
        p.max_batch_size = dataset_info.batch_size_search
    for p in problems_individual:
        p.max_batch_size = dataset_info.batch_size_search
    for p in problems_complex:
        p.max_batch_size = dataset_info.batch_size_search

    # Base results directory
    base_results_dir = os.path.join(
        current_path, "experiment_files", "search_results_compare_budget",
        str(dataset), dataset_info.transform_seq_name, str(architecture)
    )
    os.makedirs(base_results_dir, exist_ok=True)

    assert len(problems) == len(problem_names), "Mismatch between problems and problem_names."

    # Mapping from complex variants to their base algorithms
    algo_variant_mapping = {
        "pgd_c": "pgd",
        "shgo_c": "shgo",
        "pgd_restart_c": "pgd_restart",
        "shgo_c_3": "shgo",  # NEW: map to base shgo
        "pgd_c_3": "pgd",  # NEW: map to base pgd
        "pgd_vector": "pgd",
        "shgo_vector": "shgo",
    }

    # Create subsampled validation loaders (before budget loop)
    val_dataset_original = val_loader_transformed.dataset
    n_val = len(val_dataset_original)
    rng_seed = 42
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(n_val)

    half_n = n_val // 2
    quarter_n = n_val // 4

    indices_half = perm[:half_n]
    indices_quarter = perm[:quarter_n]

    # Create half-size validation loader
    val_loader_transformed_half = DataLoader(
        Subset(val_dataset_original, indices_half),
        batch_size=max(dataset_info.batch_size // 2, 1),
        shuffle=False,
        num_workers=val_loader_transformed.num_workers,
        pin_memory=True,
        persistent_workers=getattr(val_loader_transformed, "persistent_workers", False),
    )

    # Create quarter-size validation loader
    val_loader_transformed_quarter = DataLoader(
        Subset(val_dataset_original, indices_quarter),
        batch_size=max(dataset_info.batch_size // 4, 1),
        shuffle=False,
        num_workers=val_loader_transformed.num_workers,
        pin_memory=True,
        persistent_workers=getattr(val_loader_transformed, "persistent_workers", False),
    )

    test_loader_lower_batch_size = DataLoader(
        test_loader_transformed.dataset,
        batch_size=max(dataset_info.batch_size // 2, 1),
        shuffle=False,
        num_workers=test_loader_transformed.num_workers,
        pin_memory=True,
        persistent_workers=getattr(test_loader_transformed, "persistent_workers", False),
    )

    grad_weight_orig = grad_weight
    for budget in budgets:
        print(f"\n=== Budget: {budget} ===")
        for algo in all_algos:
            # Determine validation loader and eval repeats based on budget
            if budget > 200:
                val_dataset_transformed_loader = val_loader_transformed_quarter
                repeats_for_eval = eval_repeats // 2
            elif budget > 100:
                val_dataset_transformed_loader = val_loader_transformed_half
                repeats_for_eval = (eval_repeats * 3) // 4
            else:
                val_dataset_transformed_loader = val_loader_transformed
                repeats_for_eval = eval_repeats
            if "its" in algo:
                repeats_for_eval = 1

            gc.collect()
            torch.cuda.empty_cache()
            print(f"\n--- Algorithm: {algo} ---")

            # NEW: special grad_weight override
            if algo == "shgo_c_3" or algo == "pgd_c_3":
                grad_weight = 3
            else:
                grad_weight = grad_weight_orig

            # Determine which problems to use
            if algo in complex_algos:
                current_problems = problems_complex
                current_problem_names = problem_names_complex
            elif algo in ["shgo_individual", "wcd_lat_ind"]:
                current_problems = problems_individual
                current_problem_names = problem_names_individual
            elif algo in ["pgd_vector", "shgo_vector"]:
                current_problems = problems_rotvec
                current_problem_names = problem_names_rotvec
            else:
                current_problems = problems
                current_problem_names = problem_names

            for p in current_problems:
                p.max_batch_size = dataset_info.batch_size_search

            # Map algorithm names for construction
            if algo in complex_algos:
                algo_name_for_path = algo_variant_mapping[algo]
            elif algo == "shgo_individual":
                algo_name_for_path = "shgo"
            elif algo == "wcd_lat_ind":
                algo_name_for_path = "wcd_lattice"
            elif algo in ["pgd_vector", "shgo_vector"]:
                algo_name_for_path = algo_variant_mapping[algo]
            else:
                algo_name_for_path = algo

            algo_dir = os.path.join(base_results_dir, algo, f"budget_{budget}")
            os.makedirs(algo_dir, exist_ok=True)

            param_path = os.path.join(algo_dir, "best.yml")
            print(f"Result directory: {algo_dir}")

            # Load stored params or optimize
            stored_params = load_params(param_path) if os.path.exists(param_path) else None
            if stored_params is None and (algo != "cd" and "random_search" not in algo):
                default_params_kwargs = {}
                if algo in grad_weight_algos:
                    default_params_kwargs["grad_weight"] = grad_weight
                default_params = get_default_params(algo_name_for_path, budget, **default_params_kwargs)
                print("Default params (config):", default_params)

                objective_kwargs = {}
                if algo in grad_weight_algos:
                    objective_kwargs["grad_weight"] = grad_weight

                objective = make_search_objective(
                    algo=algo_name_for_path,
                    model=model,
                    val_loader=val_dataset_transformed_loader,  # Use budget-dependent validation loader
                    problem=current_problems,  # multi-problem objective
                    budget=budget,
                    device=str(device),
                    repeats=1,
                    **objective_kwargs,
                )

                pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
                study = optuna.create_study(direction="maximize", pruner=pruner)
                study.enqueue_trial(default_params)

                study.optimize(objective, n_trials=default_trials, show_progress_bar=False)

                print(f"[{algo}] Best validation value:", study.best_value)
                print("Suggested params:", study.best_trial.params)
                print("Full params:", study.best_trial.user_attrs.get("full_params"))

                save_best_trial_params(study, algo=algo_name_for_path, path=param_path)
                stored_params = load_params(param_path)
                print("Saved best params to:", param_path)
            else:
                print("Found stored best params, skipping optimization.")
                if "cd" == algo:
                    print("Note: 'cd' algorithm requires no hyperparameter optimization.")
                    stored_params = {}

            # SHGO-only sanity check: force full budget consumption by topping up n_init
            if algo in ["shgo", "shgo_individual", "shgo_c", "shgo_c_3"]:  # NEW: added shgo_c_3
                gw = stored_params.get("grad_weight", 2)
                cost = _cost_shgo(
                    stored_params["shgo_initial_samples"],
                    stored_params["shgo_local_runs"],
                    stored_params["shgo_local_steps"],
                    gw,
                )
                if cost != budget:
                    delta = budget - cost
                    # Adjust n_init up/down to match budget exactly
                    stored_params["shgo_initial_samples"] = max(
                        1,
                        stored_params["shgo_initial_samples"] + delta
                    )
                    # Recompute and assert
                    new_cost = _cost_shgo(
                        stored_params["shgo_initial_samples"],
                        stored_params["shgo_local_runs"],
                        stored_params["shgo_local_steps"],
                        gw,
                    )
                    print(f"[{algo}] adjusted n_init by {delta} to match budget: {cost} -> {new_cost}")
                    assert new_cost == budget, f"SHGO cost mismatch after fix: {new_cost}!={budget}"
                    # Persist the fix
                    save_params(stored_params, param_path)

            # assert that grad weight matches
            if algo in grad_weight_algos:
                assert stored_params.get("grad_weight", None) == grad_weight, \
                    f"Grad weight mismatch for {algo}: {stored_params.get('grad_weight', None)}!={grad_weight}"

            # Rebuild optimizer with best params
            search_obj = build_search_algorithm(
                algo_name_for_path,
                stored_params,
                problem=current_problems[0],  # any problem is fine for optimizer construction
                budget=budget,
                model=model,
            )
            print("Rebuilt search object from saved params.")

            # Evaluate per problem with cached runner
            for prob, method_name in zip(current_problems, current_problem_names):
                eval_path = os.path.join(algo_dir, f"eval_{method_name}.json")
                print(f"[{method_name}] evaluating (cached path: {eval_path})")
                metrics = load_or_run_evaluate_confidence_and_search(
                    model=model,
                    optimizer=search_obj,
                    problem=prob,
                    test_loader=test_loader_transformed if budget < 200 else test_loader_lower_batch_size,
                    save_path=eval_path,  # auto-loads if exists; saves after run otherwise
                    max_batch_override=dataset_info.batch_size_search,
                    show_progress=show_progress,
                    repeats=repeats_for_eval,  # Use budget-dependent repeats
                    return_per_run=True,
                    overwrite=False, store_val=True
                )
                gc.collect()
                torch.cuda.empty_cache()
    # %%
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # %%

    # %%

    # %%

    # %%
    import pandas as pd
    import json
    import os
    import matplotlib.pyplot as plt

    # Collect results
    results = []

    for budget in budgets:
        for algo in all_algos:
            algo_dir = os.path.join(base_results_dir, algo, f"budget_{budget}")
            for method_name in problem_names:
                eval_path = os.path.join(algo_dir, f"eval_{method_name}.json")
                if os.path.exists(eval_path):
                    with open(eval_path, "r") as f:
                        metrics = json.load(f)
                    accuracy = metrics.get("accuracy_mean", None)
                    accuracy_se = metrics.get("accuracy_se", None)
                    results.append({
                        "Algorithm": algo,
                        "Budget": budget,
                        "Problem": method_name,
                        "Accuracy": accuracy,
                        "Accuracy_SE": accuracy_se,

                    })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # %%
    df
    # %%
    transform_seq.transformations
    # %%

    import pandas as pd
    import json
    import os
    import matplotlib.pyplot as plt

    # Collect results
    results = []

    for budget in budgets:
        for algo in all_algos:
            algo_dir = os.path.join(base_results_dir, algo, f"budget_{budget}")
            for method_name in problem_names:
                eval_path = os.path.join(algo_dir, f"eval_{method_name}.json")
                if os.path.exists(eval_path):
                    with open(eval_path, "r") as f:
                        metrics = json.load(f)

                    accuracy = metrics.get("accuracy_mean", None)
                    accuracy_se = metrics.get("accuracy_se", None)
                    accuracy_std = metrics.get("accuracy_std", None)
                    number_of_runs = metrics.get("repeats", None)
                    results.append({
                        "Algorithm": algo,
                        "Budget": budget,
                        "Problem": method_name,
                        "Accuracy": accuracy,
                        "Accuracy_SE": accuracy_se,
                        "Accuracy_STD": accuracy_std,
                        "Number_of_Runs": number_of_runs,

                    })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    df
    transform_seq.transformations
    ALGO_RENAME = {
        "cd": "CD-single",
        "cd_multi_cyclus": "CD",
        "shgo": "RS-LO",
        "shgo_individual": "RS-LO-Ind",
        "parallel_sa": "PSA",
        "evolutionary": "Evo. A",
        "pso": "PSO",
        "pgd": "PGD",
        "pgd_restart": "PGD-R",
        "pgd_window": "PGD-W",
        "its": "ITS",
        "random_search": "R. Search",
        "wcd": "WCD",
        "wcd_lattice": "WCD-Lat",
        "shgo_c": "RS-LO-C",
        "pgd_c": "PGD-C",
        "shgo_c_3": "RS-LO-C3",
        "pgd_c_3": "PGD-C3",
        "wcd_lat_ind": "WCD-Lat-Ind",
        "pgd_vector": "PGDVec",
        "shgo_vector": "RS-LO-Vec",
    }

    PROBLEM_RENAME = {
        "knn_per_class": "PC-kNN",
        "logit_energy": "Logit-Energy",
        "learned_energy": "Learned-Energy",
    }

    # do some renaming

    df_renamed = df.copy()
    # rename cd to cd_signle_cycle
    # rename cd_multi_cyclus to cd
    # rename shgo to RS-LO
    df_renamed["Algorithm"] = df["Algorithm"].replace(ALGO_RENAME)

    # reanme problem name from knn_per_class to PC-kNN
    df_renamed["Problem"] = df["Problem"].replace(PROBLEM_RENAME)
    problem_names_renamed = ["Logit-Energy", "PC-kNN", "Learned-Energy"]

    fullname_dict = {
        "CD-single": "Coordinate Descent (single cycle)",
        "CD": "Coordinate Descent",
        "RS-LO": "Random Sampling with Local Optimization",
        "ITS": "Inverse Transformation Search",
        "R. Search": "Random Search",
        "PSA": "Parallel Simulated Annealing",
        "Evo. A": "Evolutionary Algorithm",
        "PSO": "Particle Swarm Optimization",
        "PGD": "Multistart Gradient Descent",
        "PGD-R": "Multistart Gradient Descent with Restarts",
        "PGD-W": "Multistart Gradient Descent with Windowing",
        "WCD": "Weighted Coordinate Descent",
        "WCD-Lat": "Weighted Coordinate Descent with Lattice Sampling",
        "RS-LO-Ind": "Random Sampling with Local Optimization (individual rotations)",
        "RS-LO-C": "Random Sampling with Local Optimization (complex/quaternion rotations)",
        "PGD-C": "Multistart Gradient Descent (complex/quaternion rotations)",
        "RS-LO-C3": "Random Sampling with Local Optimization complex/quaternion rotations, grad weight=3",
        "PGD-C3": "Multistart Gradient Descent complex/quaternion rotations, grad weight=3",
        "WCD-Lat-Ind": "Weighted Coordinate Descent with Lattice Sampling (euler rotations)",
        "PGDVec": "Multistart Gradient Descent (rotation vector representation)",
        "RS-LO-Vec": "Random Sampling with Local Optimization (rotation vector representation)",
    }

    import matplotlib.pyplot as plt

    short_names = list(fullname_dict.keys())
    n = len(short_names)

    # Get all 20 colors from tab20
    tab20_colors = [plt.get_cmap("tab20")(i)[:3] for i in range(20)]

    # Separate even and odd indices
    even_colors = tab20_colors[::2]  # 0, 2, 4, ..., 18
    odd_colors = tab20_colors[1::2]  # 1, 3, 5, ..., 19

    # Combine: even first, then odd
    ordered_colors = even_colors + odd_colors

    # Repeat pattern if more names than 20
    if n > 20:
        repeats = (n // 20) + 1
        ordered_colors = (ordered_colors * repeats)[:n]
    else:
        ordered_colors = ordered_colors[:n]

    # Assign to names
    algorithm_colors = {name: ordered_colors[i] for i, name in enumerate(short_names)}

    figure_path = os.path.join(current_path, "experiment_files", "export", "fig", "comparision_search_datasets",
                               dataset,
                               transform_name)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    from matplotlib.patches import Patch
    from utils.eval.vis import plt_setup_latex

    W = plt_setup_latex()
    handles = [
        Patch(color=algorithm_colors[short], label=f"{short} — {fullname_dict[short]}")
        for short in fullname_dict
    ]

    # Plot legend in a separate figure
    fig, ax = plt.subplots(figsize=(W, W * 0.4))
    ax.legend(
        handles=handles,
        loc='center',
        frameon=False,
        fontsize=9
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, "comparision_search_algorithms_legend.pdf"))
    plt.savefig(os.path.join(figure_path, "comparision_search_algorithms_legend.pgf"))
    plt.show()
    # %%

    figure_path = os.path.join(current_path, "experiment_files", "export", "fig", "comparision_search_budget", dataset,
                               transform_name)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    # %% md

    # %%
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    def plot_accuracy_vs_budget_side_by_side(
            df,
            problem_names,
            algo_column="Algorithm",
            problem_column="Problem",
            x_column="Budget",
            y_column="Accuracy",
            se_column="Accuracy_SE",
            algorithm_colors=None,
            fig_width=10,
            fig_height=None,
            figure_path=None,
            save_name="accuracy_vs_budget_side_by_side"
    ):
        """
        Plots Accuracy vs Budget curves for each problem side-by-side with error bars,
        and optionally saves the figure as both PDF and PGF.

        Parameters:
            df : pd.DataFrame
                DataFrame containing results.
            problem_names : list
                List of problem names to plot (order matters).
            algo_column : str
                Column name for algorithm labels.
            problem_column : str
                Column name for problem names.
            x_column : str
                Column for x-axis values (e.g. "Budget").
            y_column : str
                Column for y-axis values (e.g. "Accuracy").
            se_column : str
                Column for standard errors.
            algorithm_colors : dict
                Optional mapping {algorithm: color}.
            fig_width, fig_height : float
                Dimensions of the overall figure.
            figure_path : str
                Path to directory for saving the figure.
            save_name : str
                Base name for output file (without extension).
        """
        # remove its with budget 8
        df = df[~((df[algo_column] == "ITS") & (df[x_column] == 8))]
        sns.set(style="whitegrid", context="paper")

        n_problems = len(problem_names)
        if fig_height is None:
            fig_height = fig_width * 0.4

        fig, axes = plt.subplots(
            1, n_problems,
            figsize=(fig_width, fig_height),
            sharey=True
        )

        if n_problems == 1:
            axes = [axes]

        for ax, problem in zip(axes, problem_names):
            problem_df = df[df[problem_column] == problem]

            for algorithm in problem_df[algo_column].unique():
                alg_df = problem_df[problem_df[algo_column] == algorithm]

                color = None
                if algorithm_colors and algorithm in algorithm_colors:
                    color = algorithm_colors[algorithm]

                ax.errorbar(
                    alg_df[x_column],
                    alg_df[y_column],
                    yerr=alg_df[se_column],
                    marker="o",
                    label=algorithm,
                    capsize=2,
                    linewidth=1.5,
                    markersize=3,
                    color=color,
                    alpha=0.8
                )

            ax.set_title(problem, fontsize=9)
            ax.set_xlabel("Budget", fontsize=8)
            if ax == axes[0]:
                ax.set_ylabel("Accuracy", fontsize=8)
            else:
                ax.set_ylabel("")
            ax.tick_params(axis='both', labelsize=7)
            ax.grid(True, alpha=0.3)
            sns.despine(ax=ax)

        # Shared legend at top
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, title="Algorithm",
            loc="lower right", ncol=len(labels) // 2,
            bbox_to_anchor=(1.0, 0.25),
            fontsize=6, title_fontsize=7
        )

        plt.tight_layout(pad=0.6, w_pad=0.3)
        plt.subplots_adjust(top=0.82)  # leave room for legend

        # Save as PDF and PGF if path provided
        if figure_path is not None:
            os.makedirs(figure_path, exist_ok=True)
            pdf_path = os.path.join(figure_path, f"{save_name}.pdf")
            pgf_path = os.path.join(figure_path, f"{save_name}.pgf")
            plt.savefig(pdf_path, bbox_inches="tight")
            plt.savefig(pgf_path, bbox_inches="tight")
            print(f"Saved figure to:\n  {pdf_path}\n  {pgf_path}")

        plt.show()

    # Example usage
    plot_accuracy_vs_budget_side_by_side(
        df=df_renamed,
        problem_names=problem_names_renamed,
        algorithm_colors=algorithm_colors,
        fig_width=W,
        figure_path=figure_path
    )

    # %%
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    sns.set(style="whitegrid", context="talk")

    # --- Step 1: Compute mean and combined SE (propagating within-problem uncertainty) ---
    def compute_combined_se(group):
        """
        Compute combined mean and SE across problems, properly propagating uncertainty.
        """
        N = len(group)  # number of problems

        # Mean across problems
        mean_accuracy = group["Accuracy"].mean()

        # Variance of each problem mean: s_i^2 / n_i
        var_each = (group["Accuracy_STD"].to_numpy() ** 2) / group["Number_of_Runs"].to_numpy()

        # Combined SE across problems (propagating uncertainty)
        se_combined = np.sqrt(np.sum(var_each)) / N

        return pd.Series({
            "Accuracy_mean": mean_accuracy,
            "Accuracy_SE": se_combined
        })

    summary_df = (
        df_renamed
        .groupby(["Algorithm", "Budget"], as_index=False)
        .apply(compute_combined_se)
        .reset_index(drop=True)
    )
    summary_df["Problem"] = "Overall"  # placeholder for consistency

    plot_accuracy_vs_budget_side_by_side(
        df=summary_df,
        problem_names=["Overall"],  # matches the column value
        algo_column="Algorithm",
        problem_column="Problem",  # <- pass the actual column name
        x_column="Budget",
        y_column="Accuracy_mean",
        se_column="Accuracy_SE",
        algorithm_colors=algorithm_colors,
        fig_width=W / 2,
        fig_height=W * 0.4,
        figure_path=figure_path,
        save_name="accuracy_vs_budget_overall"
    )

    # %%
    # load its best hyperparameter per budget and print the number of hypothesis. Similarly for wcd_lattice print the number of rounds
    import os
    import yaml
    for budget in budgets:
        for algo in ["shgo_c"]:
            algo_dir = os.path.join(base_results_dir, algo, f"budget_{budget}")
            param_path = os.path.join(algo_dir, "best.yml")
            if os.path.exists(param_path):
                with open(param_path, "r") as f:
                    params = yaml.safe_load(f)
                if algo == "shgo_c":
                    n_rounds = params.get("n_rounds", None)
                    print(f"[Budget {budget}] WCD Lattice n_rounds: {n_rounds}")
                    print(params)

    # %%
    def analyze_run_results_with_budget(results_list):
        """
        Computes summary statistics per algorithm and budget for each problem.
        Uses global min-max normalization across all budgets, runs, and algorithms.
        Frobenius distance is computed against the global best run per sample.
        """
        import numpy as np
        from collections import defaultdict

        # Apply renaming first, then group
        for entry in results_list:
            algo = ALGO_RENAME.get(entry["Algorithm"], entry["Algorithm"])
            problem_name = PROBLEM_RENAME.get(entry["Problem"], entry["Problem"])
            entry["Algorithm"] = algo
            entry["Problem"] = problem_name

        results_by_problem = defaultdict(list)
        for entry in results_list:
            results_by_problem[entry["Problem"]].append(entry)

        problem_summaries = {}

        for problem, entries in results_by_problem.items():
            # First pass: collect all errors globally to find min/max and best run per sample
            all_errors_global = []
            all_mats_global = []
            metadata_global = []
            all_true_labels, all_pred_labels = [], []

            for entry in entries:
                algo = entry["Algorithm"]
                problem_name = entry["Problem"]
                budget = entry["Budget"]

                res = entry["metrics"]
                runs = res.get("per_run", [res])

                for run_idx, run in enumerate(runs):
                    if "per_sample_errors" not in run or "per_sample_matrices" not in run:
                        continue

                    errs = np.array(run["per_sample_errors"], dtype=float)
                    mats = run["per_sample_matrices"]

                    all_errors_global.append(errs)
                    all_mats_global.append(mats)
                    metadata_global.append({
                        'algo': algo,
                        'budget': budget,
                        'run_idx': run_idx
                    })

                    if "per_sample_true_labels" in run:
                        all_true_labels.append(run["per_sample_true_labels"])
                    if "per_sample_pred_labels" in run:
                        all_pred_labels.append(run["per_sample_pred_labels"])

            if not all_errors_global:
                continue

            # Check label consistency
            labels_available = len(all_true_labels) == len(all_errors_global)
            if labels_available and len(all_true_labels) > 1:
                first_labels = tuple(all_true_labels[0])
                for i, labels in enumerate(all_true_labels[1:], 1):
                    if tuple(labels) != first_labels:
                        meta0 = metadata_global[0]
                        meta_i = metadata_global[i]
                        raise ValueError(
                            f"Inconsistent 'per_sample_true_labels' in problem '{problem}'. "
                            f"Budget {meta0['budget']} algorithm '{meta0['algo']}' does not match "
                            f"budget {meta_i['budget']} algorithm '{meta_i['algo']}'."
                        )

            all_errors_global = np.stack(all_errors_global)
            eps = 1e-12
            num_samples = all_errors_global.shape[1]

            # Global min/max across all budgets and algorithms
            min_errors = np.nanmin(all_errors_global, axis=0)
            max_errors = np.nanmax(all_errors_global, axis=0)
            denom = max_errors - min_errors + eps

            # Global best run per sample
            best_run_idx_per_sample = np.nanargmin(
                np.where(np.isnan(all_errors_global), np.inf, all_errors_global), axis=0
            )
            best_mats = [all_mats_global[idx][j] for j, idx in enumerate(best_run_idx_per_sample)]

            # Best predicted labels
            best_pred_labels = None
            pred_labels_available = len(all_pred_labels) == len(all_errors_global)
            if pred_labels_available:
                best_pred_labels = [
                    all_pred_labels[idx][j] for j, idx in enumerate(best_run_idx_per_sample)
                ]

            # Vectorized relative error computation
            relative_errors_global = (all_errors_global - min_errors[None, :]) / denom[None, :]

            # Pre-compute Frobenius distances
            try:
                best_mats_array = np.array(best_mats, dtype=float)
                all_mats_stacked = []
                for run_mats in all_mats_global:
                    run_mats_array = np.array(run_mats[:num_samples], dtype=float)
                    all_mats_stacked.append(run_mats_array)
                all_mats_stacked = np.array(all_mats_stacked)

                diff = all_mats_stacked - best_mats_array[None, :, ...]
                shape = diff.shape
                diff_flat = diff.reshape(shape[0], shape[1], -1)
                best_flat = best_mats_array.reshape(shape[1], -1)

                diff_norms = np.linalg.norm(diff_flat, axis=2)
                best_norms = np.linalg.norm(best_flat, axis=1)

                frobenius_distances_global = diff_norms / (best_norms[None, :] + eps)
                valid_mask = ~np.isnan(all_errors_global) & np.isfinite(frobenius_distances_global)
                frobenius_distances_global = np.where(valid_mask, frobenius_distances_global, np.nan)

            except (ValueError, TypeError):
                frobenius_distances_global = np.full((len(all_errors_global), num_samples), np.nan)
                for run_idx in range(len(all_errors_global)):
                    mats = all_mats_global[run_idx]
                    for j in range(min(num_samples, len(mats))):
                        if np.isnan(all_errors_global[run_idx, j]):
                            continue
                        try:
                            ma = np.array(mats[j], dtype=float)
                            mb = np.array(best_mats[j], dtype=float)
                            if ma.shape == mb.shape and not (np.any(np.isnan(ma)) or np.any(np.isnan(mb))):
                                frob = np.linalg.norm(ma - mb) / (np.linalg.norm(mb) + eps)
                                if np.isfinite(frob):
                                    frobenius_distances_global[run_idx, j] = frob
                        except:
                            continue

            # Second pass: organize by budget and algorithm
            summary_by_budget = defaultdict(dict)

            for entry in entries:
                algo = entry["Algorithm"]
                budget = entry["Budget"]
                res = entry["metrics"]
                runs = res.get("per_run", [res])

                # Find indices for this algo+budget combination
                indices = [i for i, meta in enumerate(metadata_global)
                           if meta['algo'] == algo and meta['budget'] == budget]

                if not indices:
                    continue

                algo_rel_errs = relative_errors_global[indices]
                algo_frobs = frobenius_distances_global[indices]

                run_avg_rel_errs = np.nanmean(algo_rel_errs, axis=1)
                run_avg_rel_errs = run_avg_rel_errs[~np.isnan(run_avg_rel_errs)]

                run_avg_frobs = np.nanmean(algo_frobs, axis=1)
                run_avg_frobs = run_avg_frobs[~np.isnan(run_avg_frobs)]

                num_runs = len(run_avg_rel_errs)

                summary_by_budget[budget][algo] = {
                    "mean_relative_error": float(np.mean(run_avg_rel_errs)) if len(run_avg_rel_errs) > 0 else None,
                    "std_relative_error": float(np.std(run_avg_rel_errs, ddof=1)) if num_runs > 1 else None,
                    "mean_frobenius": float(np.mean(run_avg_frobs)) if len(run_avg_frobs) > 0 else None,
                    "std_frobenius": float(np.std(run_avg_frobs, ddof=1)) if num_runs > 1 else None,
                    "se_relative_error": float(
                        np.std(run_avg_rel_errs, ddof=1) / np.sqrt(num_runs)) if num_runs > 1 else None,
                    "se_frobenius": float(np.std(run_avg_frobs, ddof=1) / np.sqrt(num_runs)) if len(
                        run_avg_frobs) > 1 else None,
                    "num_runs": num_runs
                }

            problem_summaries[problem] = {
                "num_datapoints": num_samples,
                "budgets": dict(summary_by_budget),
                "true_labels": all_true_labels[0] if labels_available else None,
                "best_predicted_labels": best_pred_labels,
            }

        return problem_summaries

    # Collect results with budget information
    results_list = []
    for budget in budgets:
        for algo in all_algos:
            algo_dir = os.path.join(base_results_dir, algo, f"budget_{budget}")
            for method_name in problem_names:
                eval_path = os.path.join(algo_dir, f"eval_{method_name}.json")
                if os.path.exists(eval_path):
                    with open(eval_path, "r") as f:
                        metrics = json.load(f)
                    results_list.append({
                        "Algorithm": algo,
                        "Problem": method_name,
                        "Budget": budget,
                        "metrics": metrics
                    })

    # Analyze with budget information
    analysis = analyze_run_results_with_budget(results_list)

    # Print summary
    for problem, pdata in analysis.items():
        print(f"\n=== Problem: {problem} ===")
        print(f"Analyzed {pdata['num_datapoints']} datapoints")
        for budget in sorted(pdata["budgets"].keys()):
            print(f"\n  Budget: {budget}")
            for algo, stats in pdata["budgets"][budget].items():
                print(f"    {algo}: RelErr={stats['mean_relative_error']:.6f}±{stats['se_relative_error']:.6f}, "
                      f"Frob={stats['mean_frobenius']:.6f}±{stats['se_frobenius']:.6f}")
    # %%
    from collections import defaultdict

    def plot_error_metrics_vs_budget(
            analysis,
            problem_names,
            metric="mean_relative_error",
            se_key="se_relative_error",
            algorithm_colors=None,
            fig_width=10,
            fig_height=None,
            figure_path=None,
            save_name="error_vs_budget"
    ):
        """
        Plots error metrics vs budget for each problem side-by-side.

        Parameters:
            analysis : dict
                Output from analyze_run_results_with_budget
            problem_names : list
                List of problem names to plot
            metric : str
                Metric to plot (e.g., "mean_relative_error" or "mean_frobenius")
            se_key : str
                Standard error key
            algorithm_colors : dict
                Color mapping for algorithms
            fig_width, fig_height : float
                Figure dimensions
            figure_path : str
                Directory for saving
            save_name : str
                Base filename
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set(style="whitegrid", context="paper")

        n_problems = len(problem_names)
        if fig_height is None:
            fig_height = fig_width * 0.4

        fig, axes = plt.subplots(1, n_problems, figsize=(fig_width, fig_height), sharey=True)

        if n_problems == 1:
            axes = [axes]

        for ax, problem in zip(axes, problem_names):
            pdata = analysis[problem]
            budgets_data = pdata["budgets"]

            # Organize data by algorithm
            algo_data = defaultdict(lambda: {"budgets": [], "values": [], "errors": []})

            for budget in sorted(budgets_data.keys()):
                for algo, stats in budgets_data[budget].items():
                    if stats[metric] is not None:
                        algo_data[algo]["budgets"].append(budget)
                        algo_data[algo]["values"].append(stats[metric])
                        algo_data[algo]["errors"].append(stats[se_key] if stats[se_key] is not None else 0)

            # Plot each algorithm
            for algo in sorted(algo_data.keys()):
                data = algo_data[algo]
                color = algorithm_colors.get(algo) if algorithm_colors else None

                ax.errorbar(
                    data["budgets"],
                    data["values"],
                    yerr=data["errors"],
                    marker="o",
                    label=algo,
                    capsize=2,
                    linewidth=1.5,
                    markersize=3,
                    color=color,
                    alpha=0.8
                )

            ax.set_title(problem, fontsize=9)
            ax.set_xlabel("Budget", fontsize=8)
            if ax == axes[0]:
                ylabel = "Relative Error" if "relative" in metric else "Frobenius Distance"
                ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(axis='both', labelsize=7)
            ax.grid(True, alpha=0.3)
            sns.despine(ax=ax)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, title="Algorithm",
            loc="right", ncol=len(labels) // 2,
            bbox_to_anchor=(1.0, 0.7),
            fontsize=6, title_fontsize=7
        )

        plt.tight_layout(pad=0.6, w_pad=0.3)
        plt.subplots_adjust(top=0.82)

        if figure_path is not None:
            os.makedirs(figure_path, exist_ok=True)
            pdf_path = os.path.join(figure_path, f"{save_name}.pdf")
            pgf_path = os.path.join(figure_path, f"{save_name}.pgf")
            plt.savefig(pdf_path, bbox_inches="tight")
            plt.savefig(pgf_path, bbox_inches="tight")
            print(f"Saved: {pdf_path}")

        plt.show()

    # Plot relative error
    plot_error_metrics_vs_budget(
        analysis=analysis,
        problem_names=problem_names_renamed,
        metric="mean_relative_error",
        se_key="se_relative_error",
        algorithm_colors=algorithm_colors,
        fig_width=W,
        figure_path=figure_path,
        save_name="relative_error_vs_budget"
    )

    # Plot Frobenius distance
    plot_error_metrics_vs_budget(
        analysis=analysis,
        problem_names=problem_names_renamed,
        metric="mean_frobenius",
        se_key="se_frobenius",
        algorithm_colors=algorithm_colors,
        fig_width=W,
        figure_path=figure_path,
        save_name="frobenius_distance_vs_budget"
    )

    # %%
    def compute_overall_error_metrics(analysis):
        """
        Compute overall metrics across problems with proper SE propagation.
        """
        from collections import defaultdict

        # Organize by budget and algorithm
        data_by_budget_algo = defaultdict(lambda: defaultdict(list))

        for problem, pdata in analysis.items():
            for budget, algo_stats in pdata["budgets"].items():
                for algo, stats in algo_stats.items():
                    data_by_budget_algo[budget][algo].append(stats)

        # Compute means and combined SEs
        overall = defaultdict(lambda: defaultdict(dict))

        for budget, algo_dict in data_by_budget_algo.items():
            for algo, stats_list in algo_dict.items():
                N = len(stats_list)  # number of problems

                for metric_key in ["mean_relative_error", "mean_frobenius"]:
                    se_key = metric_key.replace("mean_", "se_")
                    std_key = metric_key.replace("mean_", "std_")

                    values = [s[metric_key] for s in stats_list if s[metric_key] is not None]
                    stds = [s[std_key] for s in stats_list if s[std_key] is not None]
                    num_runs = [s["num_runs"] for s in stats_list]

                    if not values:
                        continue

                    mean_val = np.mean(values)

                    # Propagate uncertainty across problems
                    var_each = np.array(stds) ** 2 / np.array(num_runs)
                    se_combined = np.sqrt(np.sum(var_each)) / N

                    overall[budget][algo][metric_key] = mean_val
                    overall[budget][algo][se_key] = se_combined

        return dict(overall)

    # Compute and plot overall metrics
    overall_metrics = compute_overall_error_metrics(analysis)

    # Convert to analysis format for plotting
    overall_analysis = {
        "Overall": {
            "budgets": overall_metrics,
            "num_datapoints": analysis[list(analysis.keys())[0]]["num_datapoints"]
        }
    }

    plot_error_metrics_vs_budget(
        analysis=overall_analysis,
        problem_names=["Overall"],
        metric="mean_relative_error",
        se_key="se_relative_error",
        algorithm_colors=algorithm_colors,
        fig_width=W / 2,
        fig_height=W * 0.4,
        figure_path=figure_path,
        save_name="relative_error_vs_budget_overall"
    )

    plot_error_metrics_vs_budget(
        analysis=overall_analysis,
        problem_names=["Overall"],
        metric="mean_frobenius",
        se_key="se_frobenius",
        algorithm_colors=algorithm_colors,
        fig_width=W / 2,
        fig_height=W * 0.4,
        figure_path=figure_path,
        save_name="frobenius_distance_vs_budget_overall"
    )

    # %%
    def precompute_global_minmax(analysis, base_results_dir):
        """
        Precompute global min/max errors per problem across all budgets and algorithms.

        Returns:
            dict: {problem_name: {'min_errors': array, 'max_errors': array}}
        """
        import numpy as np

        PROBLEM_RENAME_REVERSE = {v: k for k, v in PROBLEM_RENAME.items()}
        ALGO_RENAME_REVERSE = {v: k for k, v in ALGO_RENAME.items()}

        global_stats = {}

        for problem in analysis.keys():
            original_problem = PROBLEM_RENAME_REVERSE.get(problem, problem)
            pdata = analysis[problem]

            all_errors_temp = []

            # Collect all errors once per problem
            for b in pdata["budgets"].keys():
                for a in pdata["budgets"][b].keys():
                    a_orig = ALGO_RENAME_REVERSE.get(a, a)

                    a_dir = os.path.join(base_results_dir, a_orig, f"budget_{b}")
                    e_path = os.path.join(a_dir, f"eval_{original_problem}.json")

                    if os.path.exists(e_path):
                        with open(e_path, "r") as f:
                            m2 = json.load(f)
                        r2 = m2.get("per_run", [m2])
                        for r in r2:
                            if "per_sample_errors" in r:
                                all_errors_temp.append(np.array(r["per_sample_errors"], dtype=float))

            if all_errors_temp:
                all_errors_temp = np.stack(all_errors_temp)
                global_stats[problem] = {
                    'min_errors': np.nanmin(all_errors_temp, axis=0),
                    'max_errors': np.nanmax(all_errors_temp, axis=0)
                }

        return global_stats

    # Precompute global min/max once
    global_minmax = precompute_global_minmax(analysis, base_results_dir)

    # %%
    def plot_error_distributions(
            analysis,
            problem_names,
            algorithms,
            budget,
            global_minmax,
            metric="relative_error",
            algorithm_colors=None,
            fig_width=10,
            fig_height=None,
            figure_path=None,
            save_name="error_distributions"
    ):
        """
        Plots distributions of per-sample relative errors for specified algorithms at a given budget.

        Parameters:
            analysis : dict
                Output from analyze_run_results_with_budget
            problem_names : list
                List of problem names to plot (can be renamed)
            algorithms : list
                List of algorithm names to compare (can be renamed)
            budget : int
                Budget value to analyze
            global_minmax : dict
                Precomputed global min/max per problem from precompute_global_minmax()
            metric : str
                "relative_error" or "frobenius"
            algorithm_colors : dict
                Color mapping for algorithms
            fig_width, fig_height : float
                Figure dimensions
            figure_path : str
                Directory for saving
            save_name : str
                Base filename
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        PROBLEM_RENAME_REVERSE = {v: k for k, v in PROBLEM_RENAME.items()}
        ALGO_RENAME_REVERSE = {v: k for k, v in ALGO_RENAME.items()}

        sns.set(style="whitegrid", context="paper")

        n_problems = len(problem_names)
        if fig_height is None:
            fig_height = fig_width * 0.5

        fig, axes = plt.subplots(1, n_problems, figsize=(fig_width, fig_height), sharey=True)

        if n_problems == 1:
            axes = [axes]

        for ax, problem in zip(axes, problem_names):
            original_problem = PROBLEM_RENAME_REVERSE.get(problem, problem)

            # Get precomputed min/max
            min_errs = global_minmax[problem]['min_errors']
            max_errs = global_minmax[problem]['max_errors']

            for algo in algorithms:
                original_algo = ALGO_RENAME_REVERSE.get(algo, algo)

                algo_dir = os.path.join(base_results_dir, original_algo, f"budget_{budget}")
                eval_path = os.path.join(algo_dir, f"eval_{original_problem}.json")

                if not os.path.exists(eval_path):
                    continue

                with open(eval_path, "r") as f:
                    metrics = json.load(f)

                runs = metrics.get("per_run", [metrics])
                all_rel_errors = []

                for run in runs:
                    if "per_sample_errors" not in run:
                        continue

                    errs = np.array(run["per_sample_errors"], dtype=float)

                    # Use precomputed min/max
                    rel_errs = (errs - min_errs) / (max_errs - min_errs + 1e-12)
                    rel_errs = rel_errs[~np.isnan(rel_errs)]
                    all_rel_errors.extend(rel_errs)

                if all_rel_errors:
                    color = algorithm_colors.get(algo) if algorithm_colors else None
                    ax.hist(all_rel_errors, bins=30, alpha=0.6, label=algo, color=color, density=True)

            ax.set_title(problem, fontsize=9)
            ax.set_xlabel("Relative Error", fontsize=8)
            if ax == axes[0]:
                ax.set_ylabel("Density", fontsize=8)
            ax.tick_params(axis='both', labelsize=7)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(fontsize=7)
            sns.despine(ax=ax)

        plt.tight_layout()

        if figure_path is not None:
            os.makedirs(figure_path, exist_ok=True)
            pdf_path = os.path.join(figure_path, f"{save_name}_budget{budget}.pdf")
            pgf_path = os.path.join(figure_path, f"{save_name}_budget{budget}.pgf")
            plt.savefig(pdf_path, bbox_inches="tight")
            plt.savefig(pgf_path, bbox_inches="tight")
            print(f"Saved: {pdf_path}")

        plt.show()

    # Plot error distributions - much faster now
    plot_error_distributions(
        analysis=analysis,
        problem_names=problem_names_renamed,
        algorithms=["WCD-Lat", "RS-LO"],
        budget=60,
        global_minmax=global_minmax,
        algorithm_colors=algorithm_colors,
        fig_width=W * 4,
        figure_path=figure_path,
        save_name="error_distribution_comparison"
    )

    # %%
    def plot_error_distributions_by_correctness(
            analysis,
            problem_names,
            algorithms,
            budget,
            global_minmax,
            algorithm_colors=None,
            fig_width=10,
            fig_height=None,
            figure_path=None,
            save_name="error_distributions_by_correctness"
    ):
        """
        Plots distributions of per-sample relative errors split by correct/incorrect predictions from each run.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import os
        import json

        PROBLEM_RENAME_REVERSE = {v: k for k, v in PROBLEM_RENAME.items()}
        ALGO_RENAME_REVERSE = {v: k for k, v in ALGO_RENAME.items()}

        sns.set(style="whitegrid", context="paper")

        n_problems = len(problem_names)
        n_algos = len(algorithms)

        if fig_height is None:
            fig_height = fig_width * 0.6 * n_algos

        fig, axes = plt.subplots(
            n_algos, n_problems,
            figsize=(fig_width, fig_height),
            sharex=True,
            sharey='row'
        )

        if n_problems == 1:
            axes = axes.reshape(-1, 1)
        if n_algos == 1:
            axes = axes.reshape(1, -1)

        for col_idx, problem in enumerate(problem_names):
            original_problem = PROBLEM_RENAME_REVERSE.get(problem, problem)
            pdata = analysis[problem]

            true_labels = pdata.get("true_labels")
            if true_labels is None:
                print(f"Warning: No label data for {problem}, skipping")
                continue

            true_labels = np.array(true_labels)
            min_errs = global_minmax[problem]['min_errors']
            max_errs = global_minmax[problem]['max_errors']

            for row_idx, algo in enumerate(algorithms):
                ax = axes[row_idx, col_idx]
                original_algo = ALGO_RENAME_REVERSE.get(algo, algo)

                algo_dir = os.path.join(base_results_dir, original_algo, f"budget_{budget}")
                eval_path = os.path.join(algo_dir, f"eval_{original_problem}.json")

                if not os.path.exists(eval_path):
                    continue

                with open(eval_path, "r") as f:
                    metrics = json.load(f)

                runs = metrics.get("per_run", [metrics])
                correct_rel_errors = []
                incorrect_rel_errors = []

                for run in runs:
                    if "per_sample_errors" not in run or "per_sample_pred_labels" not in run:
                        continue

                    errs = np.array(run["per_sample_errors"], dtype=float)
                    pred_labels = np.array(run["per_sample_pred_labels"])

                    # Compute correctness mask for THIS run
                    n_samples = min(len(errs), len(pred_labels), len(true_labels))
                    correct_mask = (true_labels[:n_samples] == pred_labels[:n_samples])

                    # Compute relative errors
                    rel_errs = (errs[:n_samples] - min_errs[:n_samples]) / (
                                max_errs[:n_samples] - min_errs[:n_samples] + 1e-12)
                    rel_errs = rel_errs[~np.isnan(rel_errs)]

                    # Split by correctness
                    correct_rel_errors.extend(rel_errs[correct_mask])
                    incorrect_rel_errors.extend(rel_errs[~correct_mask])

                    print("Run Debug:", algo, problem, "Correct:", np.sum(correct_mask), "Incorrect:",
                          np.sum(~correct_mask))

                # Distinct colors for correct/incorrect
                correct_color = "green"
                incorrect_color = "red"

                # Scale by number of samples for proper density
                total_count = len(correct_rel_errors) + len(incorrect_rel_errors)
                if total_count > 0:
                    correct_weights = np.ones_like(correct_rel_errors)  # / total_count
                    incorrect_weights = np.ones_like(incorrect_rel_errors)  # / total_count

                    if len(correct_rel_errors) > 0:
                        ax.hist(
                            correct_rel_errors, bins=30, alpha=0.6, label="Correct",
                            color=correct_color, weights=correct_weights, edgecolor='black', linewidth=1, density=True
                        )

                    if len(incorrect_rel_errors) > 0:
                        ax.hist(
                            incorrect_rel_errors, bins=30, alpha=0.6, label="Incorrect",
                            color=incorrect_color, weights=incorrect_weights, edgecolor='black', linewidth=1,
                            density=True
                        )

                if row_idx == 0:
                    ax.set_title(problem, fontsize=9)
                if col_idx == 0:
                    ax.set_ylabel(f"{algo}\nDensity", fontsize=8)
                if row_idx == n_algos - 1:
                    ax.set_xlabel("Relative Error", fontsize=8)

                ax.tick_params(axis='both', labelsize=7)
                ax.grid(True, alpha=0.3, axis='y')

                if row_idx == 0 and col_idx == 0:
                    ax.legend(fontsize=7)

                sns.despine(ax=ax)

        plt.tight_layout()

        if figure_path is not None:
            os.makedirs(figure_path, exist_ok=True)
            pdf_path = os.path.join(figure_path, f"{save_name}_budget{budget}.pdf")
            pgf_path = os.path.join(figure_path, f"{save_name}_budget{budget}.pgf")
            plt.savefig(pdf_path, bbox_inches="tight")
            plt.savefig(pgf_path, bbox_inches="tight")
            print(f"Saved: {pdf_path}")

        plt.show()

    # Plot error distributions split by correctness
    plot_error_distributions_by_correctness(
        analysis=analysis,
        problem_names=problem_names_renamed,
        algorithms=["WCD-Lat", "RS-LO"],
        budget=240,
        global_minmax=global_minmax,
        algorithm_colors=algorithm_colors,
        fig_width=W,
        fig_height=W / 2,
        figure_path=figure_path,
        save_name="error_distribution_by_correctness"
    )


def run_2_3():
    # %%
    # Here the knn value is also 1 which is not the default value.
    # %%
    import copy

    import torch
    import torch.nn as nn
    import torchvision
    import numpy as np
    from matplotlib import pyplot as plt

    from search.parallel_gradient import ParallelGradientDescent
    from utils.sampling import BatchNegativeSampler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # look for experiment files in parents
    import os

    path_found = False
    current_path = os.getcwd()
    while not path_found:
        if os.path.exists(os.path.join(current_path, "experiment_files")):
            path_found = True
            break
        current_path = os.path.dirname(current_path)

    experiment_files_path_data = os.path.join(current_path, "experiment_files", "data")
    dataset = "modelnet10"
    default_architecutre_mapping = {
        "mnist": "resnet_small",
        "bigger_mnist": "resnet_small",
        "emnist": "extended_resnet_small",
        "bigger_emnist": "bigger_extended_resnet_small",
        "coil100": "coil_resnet_small",
        "tu_berlin": "bi_lstm",
        "modelnet10": "pointnetplus",
    }

    architecture = default_architecutre_mapping[dataset]
    # %%

    # %%

    # %%
    from experiment_thesis.dataset_preperation.get_dataset import get_dataset_info, get_dataset

    dataset_info = get_dataset_info(dataset)
    dataset_dict = get_dataset(dataset_info, path=experiment_files_path_data, batch_size=dataset_info.batch_size)
    transform_name = dataset_info.transform_seq_name
    # %%

    # %%

    # %%

    dataset_dict.keys()
    dataset_train = dataset_dict['train_dataset']
    dataset_val = dataset_dict['val_dataset']
    dataset_test = dataset_dict['test_dataset']
    train_loader = dataset_dict['train_loader']
    val_loader = dataset_dict['val_loader']
    test_loader = dataset_dict['test_loader']
    n_classes = dataset_info.num_classes
    train_loader_transformed = dataset_dict['train_loader_transformed']
    val_loader_transformed = dataset_dict['val_loader_transformed']
    test_loader_transformed = dataset_dict['test_loader_transformed']
    train_loader_no_shuffle = dataset_dict['train_loader_no_shuffle']
    # %%

    # %%

    # %%
    batch_size = next(iter(train_loader))[0].shape[0]

    # %%
    from utils.eval.vis import vis_dataset

    vis_dataset(train_loader, val_loader, test_loader_transformed)
    # %%
    from experiment_thesis.main import train_and_get_model, train_or_load_energy_model
    from experiment_thesis.dataset_preperation.basic_networks import get_network
    from utils.eval.main_model import evaluate_base_model

    model_dir_path = os.path.join(current_path, "experiment_files", "models")
    embedding_cache_path = os.path.join(current_path, "experiment_files", "embedding_cache")
    # Add results dir and helper for save paths
    results_dir_path = os.path.join(current_path, "experiment_files", "results", dataset, architecture,
                                    "comparision_over_budget")
    os.makedirs(results_dir_path, exist_ok=True)

    def savepath(label: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
        return os.path.join(results_dir_path, transform_name, f"{safe}.json")

    # %%
    model = get_network(dataset_info, architecture, num_classes=n_classes).to(device)
    modelname = f"{dataset}_{architecture}"
    cache_name_train = f"{dataset}_{architecture}_embedding_cache_train"

    train_and_get_model(model, model_dir_path, modelname, train_loader, val_loader, trainer_kwargs={
        "accelerator": "auto",
        "max_epochs": dataset_info.epochs,
        "precision": "16-mixed",
    }, load_if_exists=True)

    # %%

    # %%
    model.eval().cuda()
    # %%
    from experiment_thesis.dataset_preperation.basic_networks import make_deterministic
    make_deterministic(model)

    # %%
    res = evaluate_base_model(model, test_loader_transformed, device)
    print(res)
    # %%
    res = evaluate_base_model(model, test_loader, device)
    print(res)

    # %%
    class TensorGeometricModelUnwrapper(torch.nn.Module):
        """
        Wrapper for a torch_geometric model that receives a tuple of (pos, y) as input and creates
        a Data object from it that is passed to the model.
        """

        def __init__(self):
            super(TensorGeometricModelUnwrapper, self).__init__()

        def forward(self, data):
            # pos and y are batched tensors from a DataLoader
            # need to reconstruct the original Data objects for torch_geometric models

            pos = data.pos
            batch = data.batch
            # split pos into individual tensors based on batch
            pos_list = torch.split(pos, torch.bincount(batch).tolist())
            return torch.stack(pos_list)

    # %%
    # chek if data is iamge data
    is_image_data = len(dataset_info.input_size) == 3 and dataset_info.input_size[0] in [1, 3]
    # %%
    from experiment_thesis.dataset_preperation.transformation import get_transformation_sequence_images

    transform_seq = get_transformation_sequence_images(
        name=dataset_info.transform_seq_name,
        resample_method=dataset_info.resample_method
    ).cuda()
    # %%
    from experiment_thesis.dataset_preperation.basic_networks import get_network_layer

    layer, layer_io = get_network_layer(dataset_info, architecture, 0, num_classes=None, num_rotations=8)
    # %%
    from confidence.direct.logit_based import EnergyConfidence
    from utils.transformation_problem import TransformationProblem
    from confidence.model.single_pass import SinglePassConfidence

    logit_energy = SinglePassConfidence(model, EnergyConfidence(), index=None)
    problem_energy_logits = TransformationProblem(logit_energy, transform_seq, consolidate_method="consolidate_simple")
    # test ot
    from search.shgo import SHGO
    random_search = SHGO(initial_samples=120, local_max_steps=0)

    from utils.eval.ood_performance import load_or_run_evaluate_confidence_and_search, evaluate_confidence_and_search

    # %%
    load_or_run_evaluate_confidence_and_search(
        model, optimizer=random_search, problem=problem_energy_logits,
        test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
        save_path=savepath("energy_confidence_transformed"), show_progress=True,
        repeats=1)
    # %%
    model.cuda().eval()
    # %%
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # %%
    from utils.augments import ComposeAugmentations, random_gaussian_noise, random_contrast, \
        random_gamma, random_blur_or_sharpen, build_default_augmentations
    import utils.augments

    def dec_strat(x, idd, y_true):
        out = model(x)
        eq = out.argmax(dim=-1) == y_true
        # convert to tensor where y>=0 if correct, y<0 if incorrect
        y = torch.where(eq, y_true, -1)
        return y

    from utils.augments import build_default_augmentations, small_affine_augment_2d
    from utils.sampling_strategy import GaussianSamplingStrategyLatent, TransformLatentSamplingStrategy
    import importlib
    import utils.sampling_strategy
    import utils.sampling

    importlib.reload(utils.sampling)
    from utils.sampling import BatchNegativeSampler

    energy_model2 = get_network(dataset_info, architecture, num_classes=1).to(device)

    from experiment_thesis.main import train_or_load_energy_model

    if is_image_data:
        transform_true_function = small_affine_augment_2d
        affine_augment = utils.augments.build_default_augmentations()
    else:
        transform_true_function = None
        affine_augment = None

    negative_sampling_module = BatchNegativeSampler(
        TransformLatentSamplingStrategy(
            transform_sequence=transform_seq, ), transform_true_function
        =transform_true_function, augment_function=affine_augment,
        decision_strategy=dec_strat,
    )

    energy_conf2 = train_or_load_energy_model(
        energy_model2, model_dir_path, f"{modelname}_energy2", train_loader,
        val_loader, trainer_kwargs={
            "accelerator": "auto",
            "max_epochs": dataset_info.epochs // 2,
            "precision": "16-mixed",
        }, negative_sampling_module=negative_sampling_module, load_if_exists=True)

    # %%
    model.cuda().eval()
    # %%
    from model.pointnet_plus import SAModule
    def set_deterministic_fps(model, random_start=False):
        for module in model.modules():
            if isinstance(module, SAModule):
                module.random_start = random_start
                print(f"Set random_start={random_start} for {module.__class__.__name__}")

    set_deterministic_fps(model)
    # %%
    energy_conf2.cuda().eval()
    problem_energy2 = TransformationProblem(energy_conf2, transform_seq, consolidate_method="consolidate_simple")

    # %%
    load_or_run_evaluate_confidence_and_search(
        model, optimizer=random_search, problem=problem_energy2,
        test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
        save_path=savepath("learned_energy_confidence_transformed"), show_progress=True,
        repeats=1)
    # %%
    set_deterministic_fps(energy_conf2)
    # %%
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # %%

    # %%
    from torch.utils.data import SequentialSampler
    from embedding_cache import LayerEmbeddingCache

    cache_name_train = f"{dataset}_{architecture}_{transform_name}_embedding_cache_train"

    cache_train = LayerEmbeddingCache(model, train_loader_no_shuffle,
                                      cache_dir=os.path.join(embedding_cache_path, cache_name_train))

    dual_output_model = cache_train.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True)
    embeddings_t, final_t, classes_t = cache_train.__call__(layer, capture_modes=layer_io, flatten=True)

    from utils.transformation_problem import TransformationProblem
    from confidence.model.single_pass import SinglePassConfidence
    from confidence.direct.logit_based import EnergyConfidence
    from confidence.control.split import SplitConfidence, PredictedSplitConfidence
    from confidence.unsupervised.classic.nn_pytorch import KNNConfidence, PerClassKNNConfidence

    from confidence.input_transform import InputTransformImage, PCAInputModule, RandomProjectionModule

    nn_pytorch_pretrained = PerClassKNNConfidence(metric="cosine", input_transform=None, computation_mode="triton", k=1)
    nn_pytorch_pretrained.fit(embeddings_t, classes_t)
    nn_pytorch_pretrained.cuda()

    conf_split_pretrained = PredictedSplitConfidence(nn_pytorch_pretrained, EnergyConfidence(), mult=False, b=0.0)
    conf_mod_nn_pytorch_pretrained = SinglePassConfidence(dual_output_model, conf_split_pretrained, index=1)
    problem_nn_pytorch_pretrained = TransformationProblem(conf_mod_nn_pytorch_pretrained, transform_seq,
                                                          consolidate_method="consolidate_simple")
    model.eval().cuda()
    # %%
    load_or_run_evaluate_confidence_and_search(
        model, optimizer=random_search, problem=problem_nn_pytorch_pretrained,
        test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
        save_path=savepath("knn_per_class_confidence_transformed"), show_progress=True,
        repeats=1, overwrite=False)
    # %%
    gc.collect()
    torch.cuda.empty_cache()
    # %%
    # test wether model is deterministic
    # %%
    make_deterministic(model)
    make_deterministic(energy_conf2)
    # %%
    x = next(iter(train_loader_no_shuffle))
    # %%
    res1 = dual_output_model(x[0].to(device))[0].cpu().detach().numpy()
    # %%
    res2 = embeddings_t[:x[0].shape[0]].cpu().detach().numpy()

    # %%
    res1
    # %%
    if not np.allclose(res1, res2, atol=1e-5, rtol=1e-5):
        raise ValueError("Model is not deterministic!")
    else:
        print("Model is deterministic.")
    # %%
    from tqdm import tqdm
    import numpy as np
    import torch
    from typing import Dict

    def confidence_percentiles_from_dataloader(
            problem: "TransformationProblem",
            dataloader: "torch.utils.data.DataLoader",
            device: torch.device,
            percentile: int = 95,
            additional_classifer: torch.nn.Module = None,
    ) -> Dict[int, Dict[str, any]]:
        """
        Compute per-class confidence percentiles using the problem's error function,
        with zero parameters, but only for correctly classified samples.

        Args:
            problem: the TransformationProblem instance (must have confidence_module(x)).
            dataloader: yields batches of (x, y).
            device: torch device to put tensors on.

        Returns:
            Dict[label, {"count": int, "percentile": float}]
        """
        all_conf = []
        all_labels = []

        for batch in tqdm(dataloader, desc="Computing confidence percentiles"):
            x, y = batch
            x, y = x.to(device), y.to(device)

            errors, logits = problem.confidence_module(x)
            if logits is None:
                logits = additional_classifer(x)
            if errors.ndim > 1:
                errors = errors.mean(dim=-1)

            conf = errors.detach().cpu().numpy()  # confidence = -error
            preds = torch.argmax(logits, dim=-1)

            mask = preds.eq(y)  # only keep correct predictions

            if mask.any():
                all_conf.append(conf[mask.cpu().numpy()])
                all_labels.append(y[mask].detach().cpu().numpy())

        if not all_conf:  # in case no correct predictions
            return {}

        all_conf = np.concatenate(all_conf)
        all_labels = np.concatenate(all_labels)

        out: Dict[int, Dict[str, any]] = {}
        for c in np.unique(all_labels):
            mask = all_labels == c
            if not mask.any():
                continue
            vals = all_conf[mask]
            out[c] = {
                "count": int(mask.sum()),
                "percentile": float(np.percentile(vals, percentile)),
            }

        # return list of percentiles sorted by class
        return [out[c]["percentile"] for c in sorted(out.keys())]

    # %%
    pytorch_nn_threshold = confidence_percentiles_from_dataloader(
        problem_nn_pytorch_pretrained, train_loader_transformed, device, percentile=75)
    # %%
    energy_threshold = confidence_percentiles_from_dataloader(
        problem_energy_logits, train_loader_transformed, device, percentile=75)

    # %%
    energy_learned_threshold = confidence_percentiles_from_dataloader(
        problem_energy2, train_loader_transformed, device, percentile=75, additional_classifer=model)
    # %%
    import torch
    from typing import List, Any, Optional, Tuple, Dict

    class AdaptiveBatchMetaOptimizer:
        """
        Streams a large dataset through internal optimizers using a limited working batch.
        Tracks per-sample cycles and stops when confidence threshold or max_cycles is reached.
        """

        def __init__(
                self,
                internal_optimizers: List[Any],
                internal_batch_size: int = 128,
                confidence_threshold: float = 0.9,
                max_cycles: int = 1000,  # per-sample max optimizer calls
                device: Optional[torch.device] = None,
                retain_history: bool = False,
                global_call_cap: Optional[int] = None  # optional safety cap over ALL calls
        ):
            self.internal_optimizers = internal_optimizers
            self.internal_batch_size = internal_batch_size
            if isinstance(confidence_threshold, (float, int)):
                self.confidence_threshold = torch.tensor([confidence_threshold])
            else:
                self.confidence_threshold = torch.tensor(confidence_threshold, dtype=torch.float32)
            self.max_cycles = max_cycles
            self.device = device
            self.retain_history = retain_history
            self.global_call_cap = global_call_cap

            # Internal cumulative history
            if retain_history:
                self._history: Dict[str, list] = {
                    "n_samples": [],
                    "percentage_max_calls": []
                }
            else:
                self._history = None

        @staticmethod
        def _compute_confidence(error: torch.Tensor) -> torch.Tensor:
            return -error

        def _refill_active(self, active: torch.Tensor, queue: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            need = self.internal_batch_size - active.numel()
            if need > 0 and queue.numel() > 0:
                take = queue[:need]
                queue = queue[need:]
                if active.numel() == 0:
                    return take, queue
                active = torch.cat([active, take], dim=0)
            return active, queue

        def optimize(
                self,
                problem: Any,
                x: torch.Tensor,
                y: Optional[torch.Tensor] = None,
                verbose: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            device = self.device or x.device
            x = x.to(device)
            if y is not None:
                y = y.to(device)

            N = x.shape[0]
            best_param: Optional[torch.Tensor] = None
            best_error = torch.full((N,), float("inf"), device=device)
            best_other: Optional[torch.Tensor] = None  # stores logits / "other" info

            indices = torch.arange(N, device=device)
            active = indices[: self.internal_batch_size]
            queue = indices[self.internal_batch_size:]

            # Cycles per sample, same size as total data
            cycles = torch.zeros(N, dtype=torch.int32, device=device)

            dim_placeholder = None
            outer_pass = 0

            threshold_tensor = self.confidence_threshold.to(device)

            while active.numel() > 0:
                outer_pass += 1
                if verbose:
                    print(f"[Pass {outer_pass}] Active={active.numel()} Queue={queue.numel()} "
                          f"Remaining={active.numel() + queue.numel()}")
                    print(f"Active indices: {active.tolist()}")

                for opt in self.internal_optimizers:
                    if active.numel() == 0:
                        break
                    if self.global_call_cap is not None and cycles.sum().item() >= self.global_call_cap:
                        if verbose:
                            print("Global call cap reached.")
                        break

                    # Drop over-budget samples
                    over_budget_mask = cycles[active] >= self.max_cycles
                    if over_budget_mask.any():
                        active = active[~over_budget_mask]
                        active, queue = self._refill_active(active, queue)
                        if active.numel() == 0:
                            break

                    x_act = x[active]
                    y_act = y[active] if y is not None else None

                    seed_params = None
                    if best_param is not None:
                        seed_params = best_param[active].clone()
                        invalid = best_error[active].isinf()
                        if invalid.any():
                            seed_params[invalid] = 0.0

                    opt_kwargs = {}
                    if seed_params is not None and "initial_params" in opt.optimize.__code__.co_varnames:
                        opt_kwargs["initial_params"] = seed_params

                    # Pass y if available
                    if y_act is not None:
                        result = opt.optimize(problem, x_act, y=y_act, verbose=False, **opt_kwargs)
                    else:
                        result = opt.optimize(problem, x_act, verbose=False, **opt_kwargs)

                    if not (isinstance(result, (tuple, list)) and len(result) == 3):
                        raise RuntimeError("Internal optimizer must return (param, error, other).")

                    p_new, e_new, o_new = result

                    if dim_placeholder is None:
                        dim_placeholder = p_new.shape[-1]
                        best_param = torch.zeros((N, dim_placeholder), device=device, dtype=p_new.dtype)
                    if (o_new is not None) and (best_other is None):
                        best_other = torch.zeros((N, o_new.shape[-1]), device=device, dtype=o_new.dtype)

                    improved = e_new < best_error[active]
                    if improved.any():
                        best_param[active[improved]] = p_new[improved]
                        best_error[active[improved]] = e_new[improved]
                        if best_other is not None and o_new is not None:
                            best_other[active[improved]] = o_new[improved]

                    # Increment cycles for active batch
                    cycles[active] += 1

                    conf_active = self._compute_confidence(best_error[active])

                    # Per-class threshold using predicted class from best_other
                    if threshold_tensor.numel() == 1 or best_other is None:
                        done_conf_mask = conf_active >= threshold_tensor.item()
                    else:
                        pred_class = best_other[active].argmax(dim=-1)
                        done_conf_mask = conf_active >= threshold_tensor[pred_class]

                    done_budget_mask = cycles[active] >= self.max_cycles
                    done_mask = done_conf_mask | done_budget_mask

                    if done_mask.any():
                        keep = active[~done_mask]
                        active = keep
                        active, queue = self._refill_active(active, queue)

                    if (active.numel() == 0 and queue.numel() == 0) or \
                            (self.global_call_cap is not None and cycles.sum().item() >= self.global_call_cap):
                        break

                    # Force refill if all active exhausted budget
                    if active.numel() > 0 and (cycles[active] >= self.max_cycles).all():
                        empty = active.new_empty(0)
                        active, queue = self._refill_active(empty, queue)

            # --- Update cumulative history once per optimize() call ---
            if self.retain_history:
                n_samples = N
                percentage_used = (cycles.float() / self.max_cycles).mean().item() * 100
                self._history["n_samples"].append(n_samples)
                self._history["percentage_max_calls"].append(percentage_used)

            return best_param, best_error, best_other

        def get_and_reset_history(self) -> Optional[Tuple[Dict[str, list], float]]:
            """
            Returns the cumulative history and resets it for new runs.
            Returns a tuple:
              - hist_copy: the raw lists of n_samples and percentage_max_calls
              - weighted_avg: weighted average of percentage_max_calls over all samples
            """
            if not self.retain_history:
                return None

            hist_copy = {k: v.copy() for k, v in self._history.items()}

            if len(hist_copy["n_samples"]) > 0:
                weighted_avg = np.average(
                    hist_copy["percentage_max_calls"],
                    weights=hist_copy["n_samples"]
                )
            else:
                weighted_avg = 0.0

            # reset history
            self._history = {k: [] for k in self._history.keys()}

            return hist_copy, weighted_avg

    # %%
    import search

    di_small = search.shgo.SHGO(selection_method="topk", initial_samples=20, local_runs=1, local_max_steps=0)
    adaptive_shgo_knn = AdaptiveBatchMetaOptimizer(internal_optimizers=[di_small],
                                                   internal_batch_size=dataset_info.batch_size_search,
                                                   max_cycles=6, confidence_threshold=pytorch_nn_threshold,
                                                   retain_history=True, device=device
                                                   )
    adaptive_shgo_energy = AdaptiveBatchMetaOptimizer(internal_optimizers=[di_small],
                                                      internal_batch_size=dataset_info.batch_size_search,
                                                      max_cycles=6, confidence_threshold=energy_threshold,
                                                      retain_history=True, device=device
                                                      )
    adaptive_shgo_learned_energy = AdaptiveBatchMetaOptimizer(internal_optimizers=[di_small],
                                                              internal_batch_size=dataset_info.batch_size_search,
                                                              max_cycles=6,
                                                              confidence_threshold=energy_learned_threshold,
                                                              retain_history=True, device=device)

    total_samples = di_small.initial_samples * adaptive_shgo_knn.max_cycles
    # %%
    larger_loader_test = torch.utils.data.DataLoader(
        test_loader_transformed.dataset,
        batch_size=dataset_info.batch_size_search * 8,
        num_workers=test_loader_transformed.num_workers,
        persistent_workers=test_loader_transformed.persistent_workers, shuffle=True
    )
    # %%
    import math, json

    # === Adaptive SHGO for PyTorch NN ===
    res_adaptive_nn = load_or_run_evaluate_confidence_and_search(
        model, optimizer=adaptive_shgo_knn, problem=problem_nn_pytorch_pretrained,
        test_loader=larger_loader_test, max_batch_override=dataset_info.batch_size_search,
        save_path=savepath("knn_per_class_confidence_transformed_adaptive"), show_progress=True,
        repeats=4, overwrite=False
    )
    hist_nn, avg_nn = adaptive_shgo_knn.get_and_reset_history()

    if hist_nn is None or len(hist_nn["n_samples"]) == 0:
        import json
        with open(savepath("knn_per_class_confidence_transformed_adaptive_budget_stats"), "r") as f:
            data = json.load(f)
            hist_nn = data["history"]
            avg_nn = data["average_percentage"]
    else:
        with open(savepath("knn_per_class_confidence_transformed_adaptive_budget_stats"), "w") as f:
            json.dump({"history": hist_nn, "average_percentage": avg_nn}, f)

    budget_nn = math.ceil(total_samples * avg_nn / 100)
    shgo_budget_nn = search.shgo.SHGO(selection_method="topk", initial_samples=budget_nn,
                                      local_runs=1, local_max_steps=0)
    res_budget_nn = load_or_run_evaluate_confidence_and_search(
        model, optimizer=shgo_budget_nn, problem=problem_nn_pytorch_pretrained,
        test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
        save_path=savepath("knn_per_class_confidence_transformed_shgo_budget"), show_progress=True,
        repeats=4, overwrite=False
    )

    # === Adaptive SHGO for Energy ===
    res_adaptive_energy = load_or_run_evaluate_confidence_and_search(
        model, optimizer=adaptive_shgo_energy, problem=problem_energy_logits,
        test_loader=larger_loader_test, max_batch_override=dataset_info.batch_size_search,
        save_path=savepath("energy_per_class_confidence_transformed_adaptive"), show_progress=True,
        repeats=4, overwrite=False
    )
    hist_energy, avg_energy = adaptive_shgo_energy.get_and_reset_history()

    if hist_energy is None or len(hist_energy["n_samples"]) == 0:
        with open(savepath("energy_per_class_confidence_transformed_adaptive_budget_stats"), "r") as f:
            data = json.load(f)
            hist_energy = data["history"]
            avg_energy = data["average_percentage"]
    else:
        with open(savepath("energy_per_class_confidence_transformed_adaptive_budget_stats"), "w") as f:
            json.dump({"history": hist_energy, "average_percentage": avg_energy}, f)

    budget_energy = math.ceil(total_samples * avg_energy / 100)
    shgo_budget_energy = search.shgo.SHGO(selection_method="topk", initial_samples=budget_energy,
                                          local_runs=1, local_max_steps=0)
    res_budget_energy = load_or_run_evaluate_confidence_and_search(
        model, optimizer=shgo_budget_energy, problem=problem_energy_logits,
        test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
        save_path=savepath("energy_per_class_confidence_transformed_shgo_budget"), show_progress=True,
        repeats=4, overwrite=False
    )

    # === Adaptive SHGO for Learned Energy ===
    res_adaptive_learned = load_or_run_evaluate_confidence_and_search(
        model, optimizer=adaptive_shgo_learned_energy, problem=problem_energy2,
        test_loader=larger_loader_test, max_batch_override=dataset_info.batch_size_search,
        save_path=savepath("learned_energy_per_class_confidence_transformed_adaptive"), show_progress=True,
        repeats=4, overwrite=False
    )
    hist_learned, avg_learned = adaptive_shgo_learned_energy.get_and_reset_history()

    if hist_learned is None or len(hist_learned["n_samples"]) == 0:
        with open(savepath("learned_energy_per_class_confidence_transformed_adaptive_budget_stats"), "r") as f:
            data = json.load(f)
            hist_learned = data["history"]
            avg_learned = data["average_percentage"]
    else:
        with open(savepath("learned_energy_per_class_confidence_transformed_adaptive_budget_stats"), "w") as f:
            json.dump({"history": hist_learned, "average_percentage": avg_learned}, f)

    budget_learned = math.ceil(total_samples * avg_learned / 100)
    shgo_budget_learned = search.shgo.SHGO(selection_method="topk", initial_samples=budget_learned,
                                           local_runs=1, local_max_steps=0)
    res_budget_learned = load_or_run_evaluate_confidence_and_search(
        model, optimizer=shgo_budget_learned, problem=problem_energy2,
        test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
        save_path=savepath("learned_energy_per_class_confidence_transformed_shgo_budget"), show_progress=True,
        repeats=4, overwrite=False
    )

    # %%

    # %%
    import numpy as np
    import matplotlib.pyplot as plt

    labels = ['Adaptive SHGO', 'SHGO with Budget']
    x = np.arange(len(labels))
    width = 0.35

    # Problems already defined earlier in your code
    problems = [
        ("PyTorch NN", res_adaptive_nn, res_budget_nn, budget_nn),
        ("Energy", res_adaptive_energy, res_budget_energy, budget_energy),
        ("Learned Energy", res_adaptive_learned, res_budget_learned, budget_learned)
    ]

    # Compute global y-limits from accuracy means/errors
    all_means = []
    all_errors = []
    for _, res_adaptive, res_budget, _ in problems:
        all_means.extend([res_adaptive["accuracy_mean"], res_budget["accuracy_mean"]])
        all_errors.extend([res_adaptive["accuracy_se"], res_budget["accuracy_se"]])

    min_y = max(0, min(all_means) - max(all_errors) - 0.03)
    max_y = min(1, max(all_means) + max(all_errors) + 0.15)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (title, res_adaptive, res_budget, budget) in zip(axes, problems):
        # Ensure budget is a single integer sample count (not percent or array)
        try:
            budget_val = int(np.asarray(budget).item())
        except Exception:
            # fallback: if budget is a percentage, convert to samples (UNLIKELY per your message)
            budget_val = int(budget)

        means = [res_adaptive["accuracy_mean"], res_budget["accuracy_mean"]]
        errors = [res_adaptive["accuracy_se"], res_budget["accuracy_se"]]

        bars = ax.bar(x, means, width, yerr=errors, capsize=5, color=['skyblue', 'lightgreen'])

        # Annotate bars with accuracy mean±se
        for bar, mean, se in zip(bars, means, errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                    f"{mean:.3f}±{se:.3f}",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Put the budget in the title (explicitly show sample count)
        ax.set_title(f"{title}\nBudget = {budget_val} samples", fontsize=13, fontweight='bold')

        # Also print a small anchored label inside the axes for clarity
        ax.text(0.98, 0.98, f"budget: {budget_val}", transform=ax.transAxes,
                ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim([min_y, max_y])

    plt.tight_layout()
    plt.show()

    # %%

    # %%

    # %%

    # %%

    # %%

    # %%
    # store budget
    # %%

    # %%

    # %%
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # %%

def run_a_opt_hyper():
    # %%
    import copy
    import gc

    import torch
    import torch.nn as nn
    import torchvision
    import numpy as np
    from matplotlib import pyplot as plt

    from its.search import InverseTransformationSearch
    from search.parallel_gradient import ParallelGradientDescent
    from utils.sampling import BatchNegativeSampler

    # torch.cuda.is_available = lambda: False
    # device = torch.device("cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # look for experiment files in parents
    import os

    path_found = False
    current_path = os.getcwd()
    while not path_found:
        if os.path.exists(os.path.join(current_path, "experiment_files")):
            path_found = True
            break
        current_path = os.path.dirname(current_path)

    experiment_files_path_data = os.path.join(current_path, "experiment_files", "data")
    dataset = "modelnet10"

    default_architecutre_mapping = {
        "mnist": "resnet_small",
        "bigger_mnist": "resnet_small",
        "emnist": "extended_resnet_small",
        "bigger_emnist": "bigger_extended_resnet_small",
        "coil100": "coil_resnet_small",
        "tu_berlin": "bi_lstm",
        "modelnet10": "pointnetplus",
    }

    architecture = default_architecutre_mapping[dataset]
    budget = 60
    # %%

    # %%

    # %%
    # NOTE already rerun for using the whole embedding cache
    # %%
    from experiment_thesis.dataset_preperation.get_dataset import get_dataset_info, get_dataset

    dataset_info = get_dataset_info(dataset)
    dataset_dict = get_dataset(dataset_info, path=experiment_files_path_data, batch_size=dataset_info.batch_size)
    transform_name = dataset_info.transform_seq_name
    # %%

    # %%

    dataset_dict.keys()
    dataset_train = dataset_dict['train_dataset']
    dataset_val = dataset_dict['val_dataset']
    dataset_test = dataset_dict['test_dataset']
    train_loader = dataset_dict['train_loader']
    val_loader = dataset_dict['val_loader']
    test_loader = dataset_dict['test_loader']
    n_classes = dataset_info.num_classes
    train_loader_transformed = dataset_dict['train_loader_transformed']
    val_loader_transformed = dataset_dict['val_loader_transformed']
    test_loader_transformed = dataset_dict['test_loader_transformed']
    train_loader_no_shuffle = dataset_dict['train_loader_no_shuffle']
    # %%
    from utils.eval.vis import vis_dataset
    batch_size = next(iter(train_loader))[0].shape[0]
    vis_dataset(train_loader, val_loader, test_loader_transformed)
    # %%
    from experiment_thesis.main import train_and_get_model, train_or_load_energy_model
    from experiment_thesis.dataset_preperation.basic_networks import get_network
    from utils.eval.main_model import evaluate_base_model

    model_dir_path = os.path.join(current_path, "experiment_files", "models")
    embedding_cache_path = os.path.join(current_path, "experiment_files", "embedding_cache")
    # Add results dir and helper for save paths
    results_dir_path = os.path.join(current_path, "experiment_files", "results", dataset, architecture, "uncertainty")
    os.makedirs(results_dir_path, exist_ok=True)

    def savepath(label: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
        return os.path.join(results_dir_path, transform_name, f"{safe}.json")

    # %%
    model = get_network(dataset_info, architecture, num_classes=n_classes).to(device)
    modelname = f"{dataset}_{architecture}"
    cache_name_train = f"{dataset}_{architecture}_embedding_cache_train"

    train_and_get_model(model, model_dir_path, modelname, train_loader, val_loader, trainer_kwargs={
        "accelerator": "auto",
        "max_epochs": dataset_info.epochs,
        "precision": "16-mixed",
    }, load_if_exists=True)

    # %%
    model.eval().to(device)
    res = evaluate_base_model(model, test_loader_transformed, device)
    print(res)
    res = evaluate_base_model(model, test_loader, device)
    print(res)
    # %%
    is_image_data = len(dataset_info.input_size) == 3 and dataset_info.input_size[0] in [1, 3]

    # %%
    from utils.transforms.apply import grid_resample
    from experiment_thesis.dataset_preperation.transformation import get_transformation_sequence_images

    transform_seq = get_transformation_sequence_images(
        name=dataset_info.transform_seq_name,
        resample_method=dataset_info.resample_method,
        init_method="sobol"
    ).to(device)
    # %%
    from embedding_cache import LayerEmbeddingCache

    cache_name_train = f"{dataset}_{architecture}_{transform_name}_embedding_cache_train"

    from torch.utils.data import SequentialSampler
    import embedding_cache
    import importlib
    importlib.reload(embedding_cache)
    from embedding_cache import LayerEmbeddingCache
    from confidence.input_transform import RandomProjectionModule

    transform_name = dataset_info.transform_seq_name

    cache_name_train = f"{dataset}_{architecture}_{transform_name}_embedding_cache_train"
    from experiment_thesis.dataset_preperation.get_dataset import get_layer_embedding_cache_config, \
        create_layer_embedding_cache
    cache_config = get_layer_embedding_cache_config(dataset, architecture, transform_name=None,
                                                    dataset_info=dataset_info)
    # %%
    cache_config
    # %%
    train_cache = create_layer_embedding_cache(model, train_loader_no_shuffle, cache_config, embedding_cache_path,
                                               device=device)

    # %%
    from search.shgo import SHGO
    random_search = SHGO(initial_samples=60, local_runs=1, local_max_steps=0)
    # %%

    # %%

    model.cuda()
    model.eval()

    # %%
    import experiment_thesis.ood.base_prepare
    experiment_thesis.ood.base_prepare.OOD_PARAM_SAMPLERS
    print("")
    # %%
    from utils.sampling_strategy import TransformLatentSamplingStrategy

    sampling_strategy = TransformLatentSamplingStrategy(transform_seq, clip_data=True)
    num_negatives = 60
    sampler = BatchNegativeSampler(sampling_strategy, number_of_negatives=num_negatives).to(device)

    # %%
    id_x_mode2, id_y_mode2 = [], []
    ood_x_mode2, ood_y_mode2 = [], []

    # set random seed (optional — remove if you want non-deterministic randomness)
    torch.manual_seed(42)

    with torch.no_grad():
        for batch in val_loader:
            x_orig, y_orig = batch
            batch_size = x_orig.size(0)
            M = 1 + num_negatives  # number of candidates per sample

            # Sample positives + negatives (DO NOT change sampler)
            modified_batch = sampler((x_orig.to(device), y_orig.to(device)))
            x_all = modified_batch[0].to(device)  # shape: (batch_size * M, C, H, W)
            y_orig = y_orig.to(device)  # shape: (batch_size,)

            # Forward pass over flattened candidates (batched for memory efficiency)
            for i in range(0, x_all.shape[0], batch_size):
                x_batch = x_all[i:i + batch_size]
                logits_batch = model(x_batch)
                if i == 0:
                    logits_all = logits_batch
                else:
                    logits_all = torch.cat((logits_all, logits_batch), dim=0)

            # Group back into per-sample candidates: (B, M, num_classes)
            logits_all = torch.stack(logits_all.chunk(M, dim=0), dim=1)
            x_all = torch.stack(x_all.chunk(M, dim=0), dim=1)

            # --- ID Selection: candidate with highest logit for the true class ---
            y_idx = y_orig.view(-1, 1, 1).expand(-1, M, 1)
            true_class_logits = logits_all.gather(2, y_idx).squeeze(2)
            id_indices = torch.argmax(true_class_logits, dim=1)  # (B,)

            # --- OOD Selection: randomly pick another candidate (not the ID one) ---
            all_indices = torch.arange(M, device=device).unsqueeze(0).repeat(batch_size, 1)
            rand_indices = torch.randint(0, M - 1, (batch_size,), device=device)
            # shift indices to avoid id_indices
            ood_indices = torch.where(rand_indices >= id_indices, rand_indices + 1, rand_indices)

            # --- Gather paired samples ---
            ar = torch.arange(batch_size, device=device)
            id_x_batch = x_all[ar, id_indices]
            ood_x_batch = x_all[ar, ood_indices]

            # Append results
            id_x_mode2.append(id_x_batch.cpu())
            id_y_mode2.append(y_orig.cpu())
            ood_x_mode2.append(ood_x_batch.cpu())
            ood_y_mode2.append(y_orig.cpu())

    # Concatenate results from all batches
    id_x_mode2 = torch.cat(id_x_mode2, dim=0)
    id_y_mode2 = torch.cat(id_y_mode2, dim=0)
    ood_x_mode2 = torch.cat(ood_x_mode2, dim=0)
    ood_y_mode2 = torch.cat(ood_y_mode2, dim=0)

    # Build datasets and loaders
    dataset_id_mode2 = torch.utils.data.TensorDataset(id_x_mode2, id_y_mode2)
    dataset_ood_mode2 = torch.utils.data.TensorDataset(ood_x_mode2, ood_y_mode2)
    loader_id_mode2 = torch.utils.data.DataLoader(dataset_id_mode2, batch_size=dataset_info.batch_size, shuffle=False)
    loader_ood_mode2 = torch.utils.data.DataLoader(dataset_ood_mode2, batch_size=dataset_info.batch_size, shuffle=False)

    # %%

    # %%

    # %%
    # plot an example from the dataset to see wether they show the same image
    # check if data is iamge data
    if is_image_data and id_x_mode2.shape[1] in [1, ]:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        for i in range(5):
            plt.subplot(2, 5, i + 1)
            plt.imshow(id_x_mode2[i].cpu().squeeze(), cmap='gray')
            plt.title(f"ID Mode 1 - {id_y_mode2[i].item()}")
            plt.axis('off')
            plt.subplot(2, 5, i + 6)
            plt.imshow(ood_x_mode2[i].cpu().squeeze(), cmap='gray')
            plt.title(f"OOD Mode 1 - {id_y_mode2[i].item()}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    # %%

    # %%
    from torch.utils.data import DataLoader, Subset

    val_dataset = val_loader_transformed.dataset  # assume indexable dataset
    n_samples = len(val_dataset)
    rng = np.random.default_rng(seed=42)
    shuffled_indices = rng.permutation(n_samples)
    val_dataset_preshuffled = Subset(val_dataset, shuffled_indices)

    val_loader_transformed_preshuffled = DataLoader(
        val_dataset_preshuffled,
        batch_size=dataset_info.batch_size,
        shuffle=False,
        num_workers=val_loader_transformed.num_workers,
        pin_memory=True,
        persistent_workers=val_loader_transformed.persistent_workers,
    )
    # %%
    n_samples_fraction = int(1 / 6 * len(val_dataset))
    shuffled_indices = rng.permutation(n_samples)[:n_samples_fraction]
    val_dataset_preshuffled_small = Subset(val_dataset, shuffled_indices)
    val_loader_transformed_preshuffled_small = DataLoader(
        val_dataset_preshuffled_small,
        batch_size=dataset_info.batch_size,
        shuffle=False,
        num_workers=val_loader_transformed.num_workers,
        pin_memory=True,
        persistent_workers=val_loader_transformed.persistent_workers,
    )

    # %%
    import os
    import json
    from datetime import datetime
    import optuna
    import numpy as np
    import pandas as pd

    from experiment_thesis.ood.base_prepare import (
        run_ood_study,
        get_default_ood_params,
        get_best_ood_params_from_study,
        create_ood_problem, run_ood_study_halving,
    )
    from utils.eval.ood_performance import load_or_run_evaluate_confidence_and_search
    from search.shgo import SHGO

    # configure
    detector = "knn_mixed"
    detector2 = "knn_trap"
    ood_objectives = ["auroc", "paired_ood_acc", "fpr95"]
    n_trials_ood = 100
    n_trials_search_small = 60
    n_trials_search_large = 30
    num_runs = 4

    # Search optimizers
    optimizer_search_eval = SHGO(initial_samples=60, local_runs=1, local_max_steps=0)  # For final evaluation
    optimizer_search_small = SHGO(initial_samples=10, local_runs=1, local_max_steps=0)
    optimizer_search_large = SHGO(initial_samples=60, local_runs=1, local_max_steps=0)

    # storage dir for metadata
    ood_studies_dir = os.path.join(
        current_path,
        "experiment_files",
        "experiment_hyperparameter_opt_ood",
        dataset,
        architecture,
        getattr(dataset_info, "transform_seq_name", "default"),
    )
    os.makedirs(ood_studies_dir, exist_ok=True)

    transform_seq_arg = transform_seq

    all_results = []

    @torch.no_grad()
    def run_experiment(
            run_idx,
            exp_name,
            id_loader,
            ood_loader,
            objectives,
            n_trials,
            val_loader_for_search,
            optimizer_for_study=optimizer_search_large,  # do NOT delete this outside object
            use_halving=False,
    ):
        """Helper to run one full experiment iteration. Memory-leak-aware version."""
        print(f"\n===== Running Experiment: {exp_name} (Run {run_idx + 1}/{num_runs}) =====")
        gc.collect()
        torch.cuda.empty_cache()

        run_dir = os.path.join(ood_studies_dir, exp_name, f"run_{run_idx}")
        os.makedirs(run_dir, exist_ok=True)

        run_results = []

        for objective in objectives:
            train_cache = create_layer_embedding_cache(
                model, train_loader_no_shuffle, cache_config, embedding_cache_path, device=device
            )
            print(f"== Processing objective={objective} ==")
            gc.collect()
            torch.cuda.empty_cache()

            eval_results_path = os.path.join(run_dir, f"{objective}_eval.json")
            params_path = os.path.join(run_dir, f"{objective}_params.json")

            # If both cached files exist, load and continue (no heavy objects created)
            if os.path.exists(eval_results_path) and os.path.exists(params_path):
                print(f"  Found cached results for objective '{objective}'. Loading...")
                with open(eval_results_path, 'r') as f:
                    metrics = json.load(f)
                with open(params_path, 'r') as f:
                    best_params = json.load(f)

                search_acc = float(metrics["accuracy_mean"])
                search_acc_std = float(metrics["accuracy_std"])
                search_acc_se = float(metrics["accuracy_se"])
                print(f"  Loaded Search Accuracy: {search_acc:.4f} (+/- {search_acc_std:.4f})")

            else:
                # --- Optimization Step ---
                print(f"  Optimizing {detector} for objective={objective}...")
                study_name = f"ood_{detector}_{objective}_{exp_name}_run{run_idx}"
                storage_path = None  # No DB persistence

                is_direct_search = objective == "search"

                # Build objective kwargs (keep references minimal)
                if is_direct_search:
                    report_fraction = 0.1
                    if val_loader_for_search == val_loader_transformed_preshuffled_small:
                        report_fraction = 0.2
                        # for coil100 we can use a larger fraction
                        if dataset == "coil100":
                            report_fraction = 0.3
                        if dataset == "tu_berlin":
                            report_fraction = 0.4

                    # NOTE: we still pass model/train_cache etc. If run_ood_study stores them internally,
                    # it could retain references — so we rely on run_ood_study to not leak. After
                    # study completes we will explicitly clear local references.
                    objective_kwargs = {
                        "optimizer": optimizer_for_study,
                        "model": model,
                        "train_cache": train_cache,
                        "val_loader": val_loader_for_search,
                        "transform_seq": transform_seq_arg,
                        "dataset_info": dataset_info,
                        "architecture": architecture,
                        "device": str(device),
                        "report_fraction": report_fraction,
                        "repeats": 1,
                    }
                else:
                    objective_kwargs = {
                        "model": model,
                        "train_cache": train_cache,
                        "id_loader": id_loader,
                        "ood_loader": ood_loader,
                        "transform_seq": transform_seq_arg,
                        "dataset_info": dataset_info,
                        "architecture": architecture,
                        "device": str(device),
                        "metric": objective,
                        "check_percent": 0.1,
                        "prune_at": 0.1,
                        "max_batches": None,
                        "show_progress": False,
                    }

                # Run optimization (may be heavy)
                if not use_halving:
                    study = run_ood_study(
                        study_name=study_name,
                        storage_path=storage_path,
                        detector_name=detector,
                        objective_type=objective,
                        objective_kwargs=objective_kwargs,
                        n_trials=n_trials,
                    )
                else:
                    study = run_ood_study_halving(
                        study_name=study_name,
                        storage_path=storage_path,
                        detector_name=detector,
                        objective_type=objective,
                        objective_kwargs=objective_kwargs,
                        n_trials=n_trials,
                    )

                # get best params or defaults
                if study is None:
                    best_params = get_default_ood_params(detector)
                    print(f"  Detector {detector} is parameterless or study skipped; using defaults.")
                else:
                    best_params = get_best_ood_params_from_study(study)
                    try:
                        print(f"  Best value for {objective}: {study.best_value}")
                    except Exception:
                        pass

                # Save best params (convert any numpy/np types to python builtins if necessary)
                with open(params_path, 'w') as f:
                    json.dump(json.loads(
                        json.dumps(best_params, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))), f,
                              indent=2)
                print(f"  Saved best params to {params_path}")

                # --- Evaluation Step ---
                print(f"  Evaluating final {detector} with best params on search task...")

                # create problem (may keep references to model etc.)
                problem = create_ood_problem(
                    detector_name=detector,
                    params=best_params,
                    model=model,
                    train_cache=train_cache,
                    transform_seq=transform_seq_arg,
                    dataset_info=dataset_info,
                    architecture=architecture,
                    device=str(device),
                )

                metrics = load_or_run_evaluate_confidence_and_search(
                    model=model,
                    optimizer=optimizer_search_eval,
                    problem=problem,
                    test_loader=test_loader_transformed,
                    save_path=eval_results_path,
                    max_batch_override=dataset_info.batch_size_search,
                    show_progress=True,
                    repeats=3,
                    return_per_run=True,  # Get aggregated metrics
                    overwrite=True,  # Overwrite eval, not opt study
                    store_val=False,
                )

                # Pull out results, convert to python floats
                search_acc = float(metrics["accuracy_mean"])
                search_acc_std = float(metrics["accuracy_std"])
                search_acc_se = float(metrics["accuracy_se"])
                print(f"  Search Accuracy: {search_acc:.4f} (+/- {search_acc_std:.4f})")

                # print current cuda memory usage
                try:
                    print(f"  CUDA Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
                except Exception:
                    pass

                study = None
                problem = None
                optimizer_eval = None
                metrics = None
                best_params = None
                gc.collect()
                torch.cuda.empty_cache()

                # After evaluation we saved eval json and params json - reload lightweight dicts
                if os.path.exists(eval_results_path) and os.path.exists(params_path):
                    with open(eval_results_path, 'r') as f:
                        metrics = json.load(f)
                    with open(params_path, 'r') as f:
                        best_params = json.load(f)
                    search_acc = float(metrics["accuracy_mean"])
                    search_acc_std = float(metrics["accuracy_std"])
                    search_acc_se = float(metrics["accuracy_se"])
                    print(f"  Loaded Search Accuracy: {search_acc:.4f} (+/- {search_acc_std:.4f})")
                else:
                    # Fallback if something unexpected happened
                    print("  Warning: evaluation files not found after evaluation. Setting metrics to 0.")
                    search_acc = 0.0
                    search_acc_std = 0.0
                    best_params = get_default_ood_params(detector)

            # Final per-objective append (keep best_params small or already json-serializable)
            run_results.append({
                "exp_name": exp_name,
                "run_idx": run_idx,
                "objective": objective,
                "search_accuracy": float(search_acc),
                "search_accuracy_std": float(search_acc_std),
                "search_accuracy_se": float(search_acc_se),
                "best_params": best_params,
            })

            # Clear heavy locals for the next objective
            best_params = None
            metrics = None
            gc.collect()
            torch.cuda.empty_cache()

        return run_results

    # --- Experiment 1: OOD tuning on val vs val_transformed ---
    for i in range(num_runs):
        results = run_experiment(
            run_idx=i,
            exp_name="val_vs_val_transformed",
            id_loader=val_loader,
            ood_loader=val_loader_transformed,
            objectives=ood_objectives,
            n_trials=n_trials_ood,
            val_loader_for_search=val_loader_transformed_preshuffled,
        )
        all_results.extend(results)

    # --- Experiment 2: OOD tuning on mode2 datasets ---
    for i in range(num_runs):
        results = run_experiment(
            run_idx=i,
            exp_name="mode2_id_vs_ood",
            id_loader=loader_id_mode2,
            ood_loader=loader_ood_mode2,
            objectives=ood_objectives,
            n_trials=n_trials_ood,
            val_loader_for_search=val_loader_transformed_preshuffled,
        )
        all_results.extend(results)
    # --- Experiment 3: Direct search optimization (small) ---
    for i in range(num_runs):
        results = run_experiment(
            run_idx=i,
            exp_name="direct_search_small",
            id_loader=None,
            ood_loader=None,
            objectives=["search"],
            n_trials=n_trials_search_small,
            val_loader_for_search=val_loader_transformed_preshuffled,
            optimizer_for_study=optimizer_search_small,
        )
        all_results.extend(results)
    # --- Experiment 4: Direct search optimization (large) ---
    for i in range(num_runs):
        results = run_experiment(
            run_idx=i,
            exp_name="direct_search_large",
            id_loader=None,
            ood_loader=None,
            objectives=["search"],
            n_trials=n_trials_search_large,
            val_loader_for_search=val_loader_transformed_preshuffled,
        )
        all_results.extend(results)

    # --- Experiment 4: Direct search optimization (large) halfing---
    for i in range(num_runs):
        results = run_experiment(
            run_idx=i,
            exp_name="direct_search_large_halving",
            id_loader=None,
            ood_loader=None,
            objectives=["search"],
            n_trials=60,
            val_loader_for_search=val_loader_transformed_preshuffled,
            use_halving=True,
        )
        all_results.extend(results)

    # restricted to small val set to save time
    for i in range(num_runs):
        results = run_experiment(
            run_idx=i,
            exp_name="direct_search_val_restricted",
            id_loader=None,
            ood_loader=None,
            objectives=["search"],
            n_trials=n_trials_search_small,
            val_loader_for_search=val_loader_transformed_preshuffled_small,
        )
        all_results.extend(results)

    # --- Experiment 6: Default parameters (no optimization, no objective) ---
    for i in range(num_runs):
        print(f"\n===== Running Experiment: default_params (Run {i + 1}/{num_runs}) =====")
        gc.collect()
        torch.cuda.empty_cache()

        run_dir = os.path.join(ood_studies_dir, "default_params", f"run_{i}")
        os.makedirs(run_dir, exist_ok=True)

        eval_results_path = os.path.join(run_dir, "default_eval.json")
        params_path = os.path.join(run_dir, "default_params.json")

        if os.path.exists(eval_results_path) and os.path.exists(params_path):
            print("  Found cached default results. Loading...")
            with open(eval_results_path, 'r') as f:
                metrics = json.load(f)
            with open(params_path, 'r') as f:
                best_params = json.load(f)
            search_acc = metrics["accuracy_mean"]
            search_acc_std = metrics["accuracy_std"]
            search_acc_se = metrics["accuracy_se"]

        else:
            # --- Use default parameters ---
            best_params = get_default_ood_params(detector)
            with open(params_path, 'w') as f:
                json.dump(best_params, f, indent=2)
            print(f"  Using default params: {best_params}")

            # --- Evaluation Step ---
            problem = create_ood_problem(
                detector_name=detector,
                params=best_params,
                model=model,
                train_cache=train_cache,
                transform_seq=transform_seq_arg,
                dataset_info=dataset_info,
                architecture=architecture,
                device=str(device),
            )

            metrics = load_or_run_evaluate_confidence_and_search(
                model=model,
                optimizer=optimizer_search_eval,
                problem=problem,
                test_loader=test_loader_transformed,
                save_path=eval_results_path,
                max_batch_override=dataset_info.batch_size_search,
                show_progress=True,
                repeats=3,
                return_per_run=True,
                overwrite=True,
                store_val=False,
            )
            search_acc = metrics["accuracy_mean"]
            search_acc_std = metrics["accuracy_std"]
            search_acc_se = metrics["accuracy_se"]

        all_results.append({
            "exp_name": "default_params",
            "run_idx": i,
            "objective": "none",  # mark explicitly that there was no optimization
            "search_accuracy": search_acc,
            "search_accuracy_std": search_acc_std,
            "search_accuracy_se": search_acc_se,
            "best_params": best_params,
        })

    print("\n\n" + "=" * 20 + " FINAL RESULTS SUMMARY " + "=" * 20)
    results_df = pd.DataFrame(all_results)

    # Primary metric: mean and std of the mean accuracies across optimization runs
    # This captures the variability from the optimization process itself
    summary = results_df.groupby(['exp_name', 'objective']).agg({
        'search_accuracy': ['mean', 'std', 'max'],
        'search_accuracy_std': 'mean'  # Also report average evaluation uncertainty
    }).reset_index()

    # Flatten column names
    summary.columns = ['exp_name', 'objective', 'mean_accuracy', 'std_across_runs', 'max_accuracy', 'mean_eval_std']

    # Keep numeric version for plotting
    summary_numeric = summary.copy()

    # Format for display
    summary_display = summary.copy()
    summary_display['mean_accuracy'] = summary_display['mean_accuracy'].apply(lambda x: f"{x:.4f}")
    summary_display['std_across_runs'] = summary_display['std_across_runs'].apply(lambda x: f"{x:.4f}")
    summary_display['max_accuracy'] = summary_display['max_accuracy'].apply(lambda x: f"{x:.4f}")
    summary_display['mean_eval_std'] = summary_display['mean_eval_std'].apply(lambda x: f"{x:.4f}")

    print("\nSummary Statistics:")
    print("- mean_accuracy: Average search accuracy across optimization runs")
    print("- std_across_runs: Std dev across optimization runs (optimization variability)")
    print("- max_accuracy: Best accuracy achieved across all optimization runs")
    print("- mean_eval_std: Average evaluation uncertainty within each run")
    print()
    print(summary_display.to_string(index=False))

    # Save detailed results
    results_path = os.path.join(ood_studies_dir, "final_experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")

    # Save summary table
    summary_path = os.path.join(ood_studies_dir, "results_summary.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"Summary table saved to: {summary_path}")

    # %%
    results_df
    # %%
    # rename experiments val_vs_val_transformed becomes val_vs_transformed
    # mode2_id_vs_ood becomes MSP_vs_transformed
    # direct_search_val_restricted becomes restricted dataset
    # direct_search_small becomes smaller budget
    # direct_search_large becomes Default Budget
    # default_params becomes No Optimization
    results_df["exp_name"] = results_df["exp_name"].replace({
        "val_vs_val_transformed": "Val vs Transformed",
        "mode2_id_vs_ood": "MSP vs transformed",
        "direct_search_val_restricted": "smaller dataset",
        "direct_search_small": "smaller budget",
        "direct_search_large": "default budget",
        "default_params": "no optimization"
    })

    # ojbective paired_ood_acc becomes Paired Acc
    results_df["objective"] = results_df["objective"].replace({
        "paired_ood_acc": "paired acc.",
        "auroc": "AUROC",
        "fpr95": "FPR95",
        "search": "search",
        "none": "none"
    })

    # %%
    results_df
    # %%
    from utils.eval.vis import plt_setup_latex
    # %%
    W = plt_setup_latex()
    # %%
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os

    def plot_per_run_results_transposed(results_df, objective_colors=None, figsize=(10, 8)):
        """
        Visualize per-run search accuracies for each experiment and objective (transposed),
        including standard error bars from 'accuracy_se'.
        """

        if objective_colors is None:
            objective_colors = {
                "AUROC": "#1f77b4",  # blue
                "FPR95": "#ff7f0e",  # orange
                "paired acc.": "#2ca02c",  # green
                "search": "#d62728",  # red
                "none": "#7f7f7f",  # gray
            }

        experiments = results_df["exp_name"].unique()
        all_objectives = results_df["objective"].unique()
        n_runs = 4
        bar_height = 0.03

        fig, ax = plt.subplots(figsize=figsize)

        current_y = 0
        y_labels_pos = []
        y_labels = []

        for exp_idx, exp in enumerate(experiments):
            exp_data = results_df[results_df["exp_name"] == exp]
            objectives_in_exp = exp_data["objective"].unique()
            n_objectives_in_exp = len(objectives_in_exp)
            exp_height = n_objectives_in_exp * n_runs * bar_height

            # Plot bars for each objective
            for obj_idx, obj in enumerate(objectives_in_exp):
                obj_data = exp_data[exp_data["objective"] == obj].sort_values("search_accuracy")
                color = objective_colors.get(obj, "#333333")
                accuracies = obj_data["search_accuracy"].values

                se_values = obj_data["search_accuracy_se"].values

                for run_idx in range(len(accuracies)):
                    y_pos = current_y + (obj_idx * n_runs * bar_height) + (run_idx * bar_height)
                    ax.barh(
                        y_pos,
                        accuracies[run_idx],
                        height=bar_height,
                        color=color,
                        alpha=0.7,
                        edgecolor='black',
                        linewidth=0.5,
                        xerr=se_values[run_idx],  # ← ADD ERROR BARS
                        error_kw=dict(ecolor='black', capsize=1.5, lw=0.6)
                    )

            # Center label
            y_labels_pos.append(current_y + exp_height / 2 - bar_height / 2)
            y_labels.append(exp)
            current_y += exp_height + 0.04  # Add vertical gap

        # Set y-axis labels
        ax.set_yticks(y_labels_pos)
        ax.set_yticklabels(y_labels)

        ax.set_xlabel("Accuracy", fontsize=12)
        ax.set_title("Per-Run Search Accuracy Distribution (Transposed)",
                     fontsize=13, fontweight='bold')
        ax.grid(axis="x", linestyle="--", alpha=0.4)

        # Legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=objective_colors[obj], alpha=0.8)
                   for obj in all_objectives if obj in objective_colors]
        ax.legend(handles, [obj for obj in all_objectives if obj in objective_colors],
                  title="Objective", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)

        # Set x-axis limits with padding
        x_min = results_df["search_accuracy"].min() - 0.01
        x_max = results_df["search_accuracy"].max() + 0.01
        ax.set_xlim(x_min, x_max)

        plt.tight_layout()
        path = os.path.join(current_path, "experiment_files", "export", "results", "hyper_opt", dataset, transform_name)
        os.makedirs(path, exist_ok=True)
        plt.savefig(path + 'per_run_distributions_transposed.png', dpi=300, bbox_inches='tight')
        plt.savefig(path + 'per_run_distributions_transposed.pgf', bbox_inches='tight')
        plt.savefig(path + 'per_run_distributions_transposed.pdf', bbox_inches='tight')
        plt.show()
        print("Plot saved as 'per_run_distributions_transposed.png'")

    # Example usage:
    plot_per_run_results_transposed(results_df, figsize=(W, W))

    # %%
    summary["max_accuracy"]
    # %%

def run_a_opt_hyper_2():
    # %%
    import copy

    import torch
    import torch.nn as nn
    import torchvision
    import numpy as np
    from matplotlib import pyplot as plt

    from its.search import InverseTransformationSearch
    from search.parallel_gradient import ParallelGradientDescent
    from utils.affine_transforms_old import AffineTransformation2D
    from utils.sampling import BatchNegativeSampler

    # torch.cuda.is_available = lambda: False
    # device = torch.device("cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # look for experiment files in parents
    import os

    path_found = False
    current_path = os.getcwd()
    while not path_found:
        if os.path.exists(os.path.join(current_path, "experiment_files")):
            path_found = True
            break
        current_path = os.path.dirname(current_path)

    experiment_files_path_data = os.path.join(current_path, "experiment_files", "data")
    dataset = "tu_berlin"

    default_architecutre_mapping = {
        "mnist": "resnet_small",
        "bigger_mnist": "resnet_small",
        "emnist": "extended_resnet_small",
        "bigger_emnist": "bigger_extended_resnet_small",
        "coil100": "coil_resnet_small",
        "tu_berlin": "bi_lstm",
        "modelnet10": "pointnetplus",
    }

    architecture = default_architecutre_mapping[dataset]

    budget = None
    # %%
    from experiment_thesis.dataset_preperation.get_dataset import get_dataset_info, get_dataset

    dataset_info = get_dataset_info(dataset)

    dataset_dict = get_dataset(dataset_info, path=experiment_files_path_data, batch_size=dataset_info.batch_size)
    transform_name = dataset_info.transform_seq_name
    # %%

    dataset_dict.keys()
    dataset_train = dataset_dict['train_dataset']
    dataset_val = dataset_dict['val_dataset']
    dataset_test = dataset_dict['test_dataset']
    train_loader = dataset_dict['train_loader']
    val_loader = dataset_dict['val_loader']
    test_loader = dataset_dict['test_loader']
    n_classes = dataset_info.num_classes
    train_loader_transformed = dataset_dict['train_loader_transformed']
    val_loader_transformed = dataset_dict['val_loader_transformed']
    test_loader_transformed = dataset_dict['test_loader_transformed']
    train_loader_no_shuffle = dataset_dict['train_loader_no_shuffle']
    # %%
    x = next(iter(test_loader_transformed))[0]

    batch_size = next(iter(train_loader))[0].shape[0]

    from utils.eval.vis import vis_dataset

    vis_dataset(train_loader, val_loader, test_loader_transformed)
    from experiment_thesis.main import train_and_get_model, train_or_load_energy_model
    from experiment_thesis.dataset_preperation.basic_networks import get_network
    from utils.eval.main_model import evaluate_base_model

    model_dir_path = os.path.join(current_path, "experiment_files", "models")
    embedding_cache_path = os.path.join(current_path, "experiment_files", "embedding_cache")
    # Add results dir and helper for save paths
    results_dir_path = os.path.join(current_path, "experiment_files", "results", dataset, architecture,
                                    "unsupervised_metrics")
    os.makedirs(results_dir_path, exist_ok=True)

    def savepath(label: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
        return os.path.join(results_dir_path, transform_name, f"{safe}.json")

    # %%
    model = get_network(dataset_info, architecture, num_classes=n_classes).to(device)
    modelname = f"{dataset}_{architecture}"
    cache_name_train = f"{dataset}_{architecture}_embedding_cache_train"

    train_and_get_model(model, model_dir_path, modelname, train_loader, val_loader, trainer_kwargs={
        "accelerator": "auto",
        "max_epochs": dataset_info.epochs,
        "precision": "16-mixed",
    }, load_if_exists=True)

    # %%
    model.eval().to(device)
    # %%
    # check main model
    res = evaluate_base_model(model, test_loader_transformed, device)
    print(res)
    res = evaluate_base_model(model, test_loader, device)
    print(res)
    # %%

    # %%
    # chek if data is iamge data
    is_image_data = len(dataset_info.input_size) == 3 and dataset_info.input_size[0] in [1, 3]
    # %%
    from utils.transforms.apply import grid_resample
    from experiment_thesis.dataset_preperation.transformation import get_transformation_sequence_images

    transform_seq = get_transformation_sequence_images(
        name=dataset_info.transform_seq_name,
        resample_method=dataset_info.resample_method,
        init_method="sobol"
    ).to(device)
    # %%
    from experiment_thesis.dataset_preperation.basic_networks import make_deterministic
    make_deterministic(model)

    # %%
    from experiment_thesis.dataset_preperation.get_dataset import get_layer_embedding_cache_config, \
        create_layer_embedding_cache
    cache_config = get_layer_embedding_cache_config(dataset, architecture, transform_name=None,
                                                    dataset_info=dataset_info)
    train_cache = create_layer_embedding_cache(model, train_loader_no_shuffle, cache_config, embedding_cache_path,
                                               device=device)

    # %%
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # %%
    import os
    import json
    import torch
    import optuna
    import gc

    from experiment_thesis.ood.base_prepare import (
        run_ood_study,
        get_default_ood_params,
        get_best_ood_params_from_study,
        create_ood_problem,
    )
    from utils.eval.ood_performance import load_or_run_evaluate_confidence_and_search
    from search.shgo import SHGO

    # ---------------- CONFIG ----------------
    detectors = ["knn", "knn_mixed", "knn_mixed_faiss", "knn_itf", "vim"]
    search_objective = "search"

    optimizer_search_eval = SHGO(initial_samples=60, local_runs=1, local_max_steps=0)
    optimizer_search_small = SHGO(initial_samples=10, local_runs=1, local_max_steps=0)
    optimizer_search_large = SHGO(initial_samples=60, local_runs=1, local_max_steps=0)

    # %%
    if dataset in ["mnist", "bigger_mnist", "emnist", "bigger_emnist"]:
        report_fraction_stage1 = 0.2
    else:
        report_fraction_stage1 = 0.4
    # %%
    import os
    import json
    import torch
    import gc
    import numpy as np
    import copy
    from torch.utils.data import DataLoader, Subset
    import optuna

    from experiment_thesis.ood.base_prepare import (
        run_ood_study,
        get_default_ood_params,
        get_best_ood_params_from_study,
        create_ood_problem,
    )
    from utils.eval.ood_performance import load_or_run_evaluate_confidence_and_search
    from search.shgo import SHGO

    search_objective = "search"

    n_trials_search = 100
    n_trials_search_refine = 10
    eval_repeats = 8
    show_progress = True
    top_k = 6
    transform_seq_arg = transform_seq

    optimizer_search_eval = SHGO(initial_samples=60, local_runs=1, local_max_steps=0)

    base_results_dir = os.path.join(
        current_path,
        "experiment_files",
        "ood_studies_v1_one_loader",
        str(dataset),
        str(architecture),
        getattr(dataset_info, "transform_seq_name", "default"),
    )
    os.makedirs(base_results_dir, exist_ok=True)

    model.eval().to(device)

    # ---------------- Prepare validation subset ----------------
    val_dataset = val_loader_transformed.dataset
    n_samples = len(val_dataset)
    subset_size = n_samples // 6

    rng = np.random.default_rng(seed=42)
    all_indices = rng.permutation(n_samples)

    val_subset_1 = Subset(val_dataset, all_indices[:subset_size])

    val_loader_small_1 = DataLoader(
        val_subset_1,
        batch_size=dataset_info.batch_size,
        shuffle=False,
        num_workers=val_loader_transformed.num_workers,
        pin_memory=True,
        persistent_workers=val_loader_transformed.persistent_workers
    )

    val_loader_transformed_preshuffled = DataLoader(
        val_dataset,
        batch_size=dataset_info.batch_size,
        shuffle=False,
        num_workers=val_loader_transformed.num_workers,
        pin_memory=True,
        persistent_workers=val_loader_transformed.persistent_workers
    )

    print(f"Each subset: {subset_size} samples ({subset_size / n_samples:.1%} of dataset)")
    print("One disjoint subset created.\n")

    # ---------------- Loop over detectors ----------------
    for detector in detectors:
        print(f"\n=== Detector (V1): {detector} ===")
        detector_dir = os.path.join(base_results_dir, detector)
        os.makedirs(detector_dir, exist_ok=True)

        params_path = os.path.join(detector_dir, "best_params.json")
        eval_path = os.path.join(detector_dir, "eval_results.json")

        # ---------------- Load or run search ----------------
        if os.path.exists(params_path):
            print(f"[{detector}] Found existing best_params.json, skipping search.")
            try:
                with open(params_path, "r") as f:
                    best_params = json.load(f)
            except Exception as e:
                print(f"[{detector}] Warning: Failed to read best_params.json ({e}), using default params.")
                best_params = get_default_ood_params(detector)
        else:
            gc.collect();
            torch.cuda.empty_cache()
            default_params = get_default_ood_params(detector)
            best_stage1_params_all = []

            val_loaders_small = [val_loader_small_1]
            trials_per_loader = [n_trials_search]

            # === Stage 1: coarse search ===
            for i, (small_loader, n_trials_this_run) in enumerate(zip(val_loaders_small, trials_per_loader), start=1):
                print(f"\n[{detector}] Running on coarse loader {i}/1 ({n_trials_this_run} trials)...")

                optimizer = SHGO(initial_samples=60, local_runs=1, local_max_steps=0)
                objective_kwargs_stage1 = {
                    "optimizer": optimizer,
                    "model": model,
                    "train_cache": train_cache,
                    "val_loader": small_loader,
                    "transform_seq": transform_seq_arg,
                    "dataset_info": dataset_info,
                    "architecture": architecture,
                    "device": str(device),
                    "report_fraction": report_fraction_stage1,
                    "repeats": 1,
                }

                study_stage1 = run_ood_study(
                    study_name=f"{detector}_v1_stage1_part{i}",
                    storage_path=None,
                    detector_name=detector,
                    objective_type=search_objective,
                    objective_kwargs=objective_kwargs_stage1,
                    n_trials=n_trials_this_run,
                    enqueue_params=[copy.deepcopy(default_params)],
                )

                if study_stage1 is not None:
                    completed_trials = [t for t in study_stage1.trials if t.state == optuna.trial.TrialState.COMPLETE]
                    if not completed_trials:
                        print(f"[{detector}] Warning: No completed trials in subset {i}")
                        continue

                    topk_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:top_k]
                    for t in topk_trials:
                        best_stage1_params_all.append(copy.deepcopy(t.params))

            # === Stage 2: refine search ===
            print(f"\n[{detector}] Stage 2: refine search...")
            optimizer_stage2 = SHGO(initial_samples=60, local_runs=1, local_max_steps=0)
            objective_kwargs_stage2 = {
                "optimizer": optimizer_stage2,
                "model": model,
                "train_cache": train_cache,
                "val_loader": val_loader_transformed_preshuffled,
                "transform_seq": transform_seq_arg,
                "dataset_info": dataset_info,
                "architecture": architecture,
                "device": str(device),
                "report_fraction": 0.1,
                "repeats": 1,
            }

            enqueue_list = [copy.deepcopy(default_params)] + [copy.deepcopy(p) for p in best_stage1_params_all]
            study_stage2 = run_ood_study(
                study_name=f"{detector}_v1_stage2",
                storage_path=None,
                detector_name=detector,
                objective_type=search_objective,
                objective_kwargs=objective_kwargs_stage2,
                n_trials=n_trials_search_refine,
                enqueue_params=enqueue_list,
            )

            best_params = (
                get_best_ood_params_from_study(study_stage2)
                if study_stage2
                else (best_stage1_params_all[-1] if best_stage1_params_all else default_params)
            )

            with open(params_path, "w") as f:
                json.dump(best_params, f, indent=2)
            print(f"[{detector}] Saved best parameters to {params_path}")

        # ---------------- Evaluate ----------------
        run_evaluation = True
        if os.path.exists(eval_path):
            try:
                with open(eval_path, "r") as f:
                    eval_data = json.load(f)
                if eval_data.get("number_of_runs", 0) >= eval_repeats:
                    print(f"[{detector}] Evaluation already complete, skipping.")
                    run_evaluation = False
            except Exception as e:
                print(f"[{detector}] Warning: Could not read eval JSON ({e}), re-running evaluation.")

        if run_evaluation:
            print(f"[{detector}] Evaluating final configuration...")
            problem = create_ood_problem(
                detector_name=detector,
                params=best_params,
                model=model,
                train_cache=train_cache,
                transform_seq=transform_seq_arg,
                dataset_info=dataset_info,
                architecture=architecture,
                device=str(device),
            )

            metrics = load_or_run_evaluate_confidence_and_search(
                model=model,
                optimizer=optimizer_search_eval,
                problem=problem,
                test_loader=test_loader_transformed,
                save_path=eval_path,
                max_batch_override=dataset_info.batch_size_search,
                show_progress=show_progress,
                repeats=eval_repeats,
                return_per_run=True,
                overwrite=False,
                store_val=False,
            )

            print(
                f"[{detector}] Final Search Accuracy (V1): "
                f"{metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}"
            )

    print("\nAll detectors processed successfully (V1).")

    # %%
    import os
    import json
    import torch
    import gc
    import numpy as np
    import copy
    from torch.utils.data import DataLoader, Subset
    import optuna

    from experiment_thesis.ood.base_prepare import (
        run_ood_study,
        get_default_ood_params,
        get_best_ood_params_from_study,
        create_ood_problem,
    )
    from utils.eval.ood_performance import load_or_run_evaluate_confidence_and_search
    from search.shgo import SHGO

    search_objective = "search"

    n_trials_search_v2 = 100
    n_trials_search_refine = 10
    eval_repeats = 8
    show_progress = True
    top_k = 3
    transform_seq_arg = transform_seq

    optimizer_search_eval = SHGO(initial_samples=60, local_runs=1, local_max_steps=0)

    base_results_dir = os.path.join(
        current_path,
        "experiment_files",
        "ood_studies_v2_two_loaders",
        str(dataset),
        str(architecture),
        getattr(dataset_info, "transform_seq_name", "default"),
    )
    os.makedirs(base_results_dir, exist_ok=True)

    model.eval().to(device)

    # ---------------- Prepare validation subsets ----------------
    val_dataset = val_loader_transformed.dataset
    n_samples = len(val_dataset)
    subset_size = n_samples // 6

    rng = np.random.default_rng(seed=42)
    all_indices = rng.permutation(n_samples)

    val_subsets = [
        Subset(val_dataset, all_indices[i * subset_size:(i + 1) * subset_size])
        for i in range(2)
    ]

    val_loaders_small = [
        DataLoader(
            subset,
            batch_size=dataset_info.batch_size,
            shuffle=False,
            num_workers=val_loader_transformed.num_workers,
            pin_memory=True,
            persistent_workers=val_loader_transformed.persistent_workers
        )
        for subset in val_subsets
    ]

    val_loader_transformed_preshuffled = DataLoader(
        val_dataset,
        batch_size=dataset_info.batch_size,
        shuffle=False,
        num_workers=val_loader_transformed.num_workers,
        pin_memory=True,
        persistent_workers=val_loader_transformed.persistent_workers
    )

    print(f"Each subset: {subset_size} samples ({subset_size / n_samples:.1%} of dataset)")
    print("Two disjoint subsets created.\n")

    # ---------------- Loop over detectors ----------------
    for detector in detectors:
        print(f"\n=== Detector (V2): {detector} ===")
        detector_dir = os.path.join(base_results_dir, detector)
        os.makedirs(detector_dir, exist_ok=True)

        params_path = os.path.join(detector_dir, "best_params.json")
        eval_path = os.path.join(detector_dir, "eval_results.json")

        # ---------------- Load or run search ----------------
        if os.path.exists(params_path):
            print(f"[{detector}] Found existing best_params.json, skipping search.")
            try:
                with open(params_path, "r") as f:
                    best_params = json.load(f)
            except Exception as e:
                print(f"[{detector}] Warning: Failed to read best_params.json ({e}), using default params.")
                best_params = get_default_ood_params(detector)
        else:
            gc.collect();
            torch.cuda.empty_cache()
            default_params = get_default_ood_params(detector)
            best_stage1_params_all = []

            trials_per_loader = [n_trials_search_v2 // 2, n_trials_search_v2 - n_trials_search_v2 // 2]

            # === Stage 1: coarse search on subsets ===
            for i, (small_loader, n_trials_this_run) in enumerate(zip(val_loaders_small, trials_per_loader), start=1):
                print(f"\n[{detector}] Running on coarse loader {i}/2 ({n_trials_this_run} trials)...")

                optimizer = SHGO(initial_samples=60, local_runs=1, local_max_steps=0)
                objective_kwargs_stage1 = {
                    "optimizer": optimizer,
                    "model": model,
                    "train_cache": train_cache,
                    "val_loader": small_loader,
                    "transform_seq": transform_seq_arg,
                    "dataset_info": dataset_info,
                    "architecture": architecture,
                    "device": str(device),
                    "report_fraction": report_fraction_stage1,
                    "repeats": 1,
                }

                study_stage1 = run_ood_study(
                    study_name=f"{detector}_v2_stage1_part{i}",
                    storage_path=None,
                    detector_name=detector,
                    objective_type=search_objective,
                    objective_kwargs=objective_kwargs_stage1,
                    n_trials=n_trials_this_run,
                    enqueue_params=[copy.deepcopy(default_params)],
                )

                if study_stage1 is not None:
                    completed_trials = [t for t in study_stage1.trials if t.state == optuna.trial.TrialState.COMPLETE]
                    if not completed_trials:
                        print(f"[{detector}] Warning: No completed trials in subset {i}")
                        continue

                    topk_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:top_k]
                    for t in topk_trials:
                        best_stage1_params_all.append(copy.deepcopy(t.params))

            # === Stage 2: refine search ===
            print(f"\n[{detector}] Stage 2: refine search...")
            optimizer_stage2 = SHGO(initial_samples=60, local_runs=1, local_max_steps=0)
            objective_kwargs_stage2 = {
                "optimizer": optimizer_stage2,
                "model": model,
                "train_cache": train_cache,
                "val_loader": val_loader_transformed_preshuffled,
                "transform_seq": transform_seq_arg,
                "dataset_info": dataset_info,
                "architecture": architecture,
                "device": str(device),
                "report_fraction": 0.1,
                "repeats": 1,
            }

            enqueue_list = [copy.deepcopy(default_params)] + [copy.deepcopy(p) for p in best_stage1_params_all]
            study_stage2 = run_ood_study(
                study_name=f"{detector}_v2_stage2",
                storage_path=None,
                detector_name=detector,
                objective_type=search_objective,
                objective_kwargs=objective_kwargs_stage2,
                n_trials=n_trials_search_refine,
                enqueue_params=enqueue_list,
            )

            best_params = (
                get_best_ood_params_from_study(study_stage2)
                if study_stage2
                else (best_stage1_params_all[-1] if best_stage1_params_all else default_params)
            )

            with open(params_path, "w") as f:
                json.dump(best_params, f, indent=2)
            print(f"[{detector}] Saved best parameters to {params_path}")

        # ---------------- Evaluate ----------------
        run_evaluation = True
        if os.path.exists(eval_path):
            try:
                with open(eval_path, "r") as f:
                    eval_data = json.load(f)
                if eval_data.get("number_of_runs", 0) >= eval_repeats:
                    print(f"[{detector}] Evaluation already complete, skipping.")
                    run_evaluation = False
            except Exception as e:
                print(f"[{detector}] Warning: Could not read eval JSON ({e}), re-running evaluation.")

        if run_evaluation:
            print(f"[{detector}] Evaluating final configuration...")
            problem = create_ood_problem(
                detector_name=detector,
                params=best_params,
                model=model,
                train_cache=train_cache,
                transform_seq=transform_seq_arg,
                dataset_info=dataset_info,
                architecture=architecture,
                device=str(device),
            )

            metrics = load_or_run_evaluate_confidence_and_search(
                model=model,
                optimizer=optimizer_search_eval,
                problem=problem,
                test_loader=test_loader_transformed,
                save_path=eval_path,
                max_batch_override=dataset_info.batch_size_search,
                show_progress=show_progress,
                repeats=eval_repeats,
                return_per_run=True,
                overwrite=False,
                store_val=False,
            )

            print(
                f"[{detector}] Final Search Accuracy (V2): "
                f"{metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}"
            )

    print("\nAll detectors processed successfully (V2).")

    # %%
    import os
    import json
    import torch
    import gc
    import numpy as np
    import copy
    from torch.utils.data import DataLoader, Subset
    import optuna

    from experiment_thesis.ood.base_prepare import (
        run_ood_study,
        get_default_ood_params,
        get_best_ood_params_from_study,
        create_ood_problem,
    )
    from utils.eval.ood_performance import load_or_run_evaluate_confidence_and_search
    from search.shgo import SHGO

    search_objective = "search"

    n_trials_search_v3 = 100
    n_trials_search_refine = 10
    eval_repeats = 8
    show_progress = True
    top_k = 2
    transform_seq_arg = transform_seq

    optimizer_search_eval = SHGO(initial_samples=60, local_runs=1, local_max_steps=0)

    base_results_dir = os.path.join(
        current_path,
        "experiment_files",
        "ood_studies_v3_three_loaders",
        str(dataset),
        str(architecture),
        getattr(dataset_info, "transform_seq_name", "default"),
    )
    os.makedirs(base_results_dir, exist_ok=True)

    model.eval().to(device)

    # ---------------- Create 3 disjoint 1/6 subsets ----------------
    val_dataset = val_loader_transformed.dataset
    n_samples = len(val_dataset)
    subset_size = n_samples // 6

    rng = np.random.default_rng(seed=42)
    all_indices = rng.permutation(n_samples)

    subsets = [
        Subset(val_dataset, all_indices[i * subset_size:(i + 1) * subset_size])
        for i in range(3)
    ]

    val_loaders_small = [
        DataLoader(
            subset,
            batch_size=dataset_info.batch_size,
            shuffle=False,
            num_workers=val_loader_transformed.num_workers,
            pin_memory=True,
            persistent_workers=val_loader_transformed.persistent_workers,
        )
        for subset in subsets
    ]

    val_loader_transformed_preshuffled = DataLoader(
        val_dataset,
        batch_size=dataset_info.batch_size,
        shuffle=False,
        num_workers=val_loader_transformed.num_workers,
        pin_memory=True,
        persistent_workers=val_loader_transformed.persistent_workers,
    )

    print(f"Each subset: {subset_size} samples ({subset_size / n_samples:.1%} of dataset)")
    print("Three disjoint subsets created.\n")

    # ---------------- Loop over detectors ----------------
    for detector in detectors:
        print(f"\n=== Detector (V3): {detector} ===")
        detector_dir = os.path.join(base_results_dir, detector)
        os.makedirs(detector_dir, exist_ok=True)
        params_path = os.path.join(detector_dir, "best_params.json")
        eval_path = os.path.join(detector_dir, "eval_results.json")

        # ---------------- Load or run search ----------------
        if os.path.exists(params_path):
            print(f"[{detector}] Found existing best_params.json, skipping search.")
            try:
                with open(params_path, "r") as f:
                    best_params = json.load(f)
            except Exception as e:
                print(f"[{detector}] Warning: Failed to read best_params.json ({e}), using default params.")
                best_params = get_default_ood_params(detector)
        else:
            # === Stage 1: coarse search on subsets ===
            gc.collect();
            torch.cuda.empty_cache()
            default_params = get_default_ood_params(detector)
            best_stage1_params_all = []

            trials_per_loader = [n_trials_search_v3 // 3] * 3
            trials_per_loader[-1] += n_trials_search_v3 - sum(trials_per_loader)

            for i, (small_loader, n_trials_this_run) in enumerate(zip(val_loaders_small, trials_per_loader), start=1):
                print(f"\n[{detector}] Coarse loader {i}/3 ({n_trials_this_run} trials)...")

                optimizer = SHGO(initial_samples=60, local_runs=1, local_max_steps=0)
                objective_kwargs_stage1 = {
                    "optimizer": optimizer,
                    "model": model,
                    "train_cache": train_cache,
                    "val_loader": small_loader,
                    "transform_seq": transform_seq_arg,
                    "dataset_info": dataset_info,
                    "architecture": architecture,
                    "device": str(device),
                    "report_fraction": report_fraction_stage1,
                    "repeats": 1,
                }

                study_stage1 = run_ood_study(
                    study_name=f"{detector}_v3_stage1_part{i}",
                    storage_path=None,
                    detector_name=detector,
                    objective_type=search_objective,
                    objective_kwargs=objective_kwargs_stage1,
                    n_trials=n_trials_this_run,
                    enqueue_params=[copy.deepcopy(default_params)],
                )

                if study_stage1 is not None:
                    completed_trials = [t for t in study_stage1.trials if t.state == optuna.trial.TrialState.COMPLETE]
                    if not completed_trials:
                        print(f"[{detector}] Warning: No completed trials in subset {i}")
                        continue

                    topk_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:top_k]
                    for t in topk_trials:
                        best_stage1_params_all.append(copy.deepcopy(t.params))

            # === Stage 2: refine search ===
            print(f"\n[{detector}] Stage 2: refine search...")
            optimizer_stage2 = SHGO(initial_samples=60, local_runs=1, local_max_steps=0)
            objective_kwargs_stage2 = {
                "optimizer": optimizer_stage2,
                "model": model,
                "train_cache": train_cache,
                "val_loader": val_loader_transformed_preshuffled,
                "transform_seq": transform_seq_arg,
                "dataset_info": dataset_info,
                "architecture": architecture,
                "device": str(device),
                "report_fraction": 0.1,
                "repeats": 1,
            }

            enqueue_list = [copy.deepcopy(default_params)] + [copy.deepcopy(p) for p in best_stage1_params_all]
            study_stage2 = run_ood_study(
                study_name=f"{detector}_v3_stage2",
                storage_path=None,
                detector_name=detector,
                objective_type=search_objective,
                objective_kwargs=objective_kwargs_stage2,
                n_trials=n_trials_search_refine,
                enqueue_params=enqueue_list,
            )

            best_params = (
                get_best_ood_params_from_study(study_stage2)
                if study_stage2
                else (best_stage1_params_all[-1] if best_stage1_params_all else default_params)
            )

            with open(params_path, "w") as f:
                json.dump(best_params, f, indent=2)
            print(f"[{detector}] Saved best parameters to {params_path}")

        # ---------------- Evaluate ----------------
        run_evaluation = True
        if os.path.exists(eval_path):
            try:
                with open(eval_path, "r") as f:
                    eval_data = json.load(f)
                if eval_data.get("number_of_runs", 0) >= eval_repeats:
                    print(f"[{detector}] Evaluation already complete, skipping.")
                    run_evaluation = False
            except Exception as e:
                print(f"[{detector}] Warning: Could not read eval JSON ({e}), re-running evaluation.")

        if run_evaluation:
            print(f"[{detector}] Evaluating best config...")
            problem = create_ood_problem(
                detector_name=detector,
                params=best_params,
                model=model,
                train_cache=train_cache,
                transform_seq=transform_seq_arg,
                dataset_info=dataset_info,
                architecture=architecture,
                device=str(device),
            )

            metrics = load_or_run_evaluate_confidence_and_search(
                model=model,
                optimizer=optimizer_search_eval,
                problem=problem,
                test_loader=test_loader_transformed,
                save_path=eval_path,
                max_batch_override=dataset_info.batch_size_search,
                show_progress=show_progress,
                repeats=eval_repeats,
                return_per_run=True,
                overwrite=False,
                store_val=False,
            )

            print(
                f"[{detector}] Final Search Accuracy (V3): "
                f"{metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}"
            )

    print("\nAll detectors processed successfully (V3).")

    # %%
    import os
    import json
    import torch
    import gc
    import copy

    from experiment_thesis.ood.base_prepare import (
        run_ood_study_halving,
        get_default_ood_params,
        get_best_ood_params_from_study,
        create_ood_problem,
    )
    from utils.eval.ood_performance import load_or_run_evaluate_confidence_and_search
    from search.shgo import SHGO

    search_objective = "search"
    n_trials_halving = 60
    eval_repeats = 8
    show_progress = True
    transform_seq_arg = transform_seq

    optimizer_search_eval = SHGO(initial_samples=60, local_runs=1, local_max_steps=0)

    base_results_dir = os.path.join(
        current_path,
        "experiment_files",
        "ood_studies_halving_full_loader",
        str(dataset),
        str(architecture),
        getattr(dataset_info, "transform_seq_name", "default"),
    )
    os.makedirs(base_results_dir, exist_ok=True)

    model.eval().to(device)

    for detector in detectors:
        print(f"\n=== Detector (Halving): {detector} ===")
        detector_dir = os.path.join(base_results_dir, detector)
        os.makedirs(detector_dir, exist_ok=True)
        params_path = os.path.join(detector_dir, "best_params.json")
        eval_path = os.path.join(detector_dir, "eval_results.json")

        # Load or run search only if best_params.json does NOT exist
        if os.path.exists(params_path):
            print(f"[{detector}] Found existing best_params.json, skipping search.")
            try:
                with open(params_path, "r") as f:
                    best_params = json.load(f)
            except Exception as e:
                print(f"[{detector}] Warning: Failed to read best_params.json ({e}), using default params.")
                best_params = get_default_ood_params(detector)
        else:
            gc.collect();
            torch.cuda.empty_cache()
            default_params = get_default_ood_params(detector)

            objective_kwargs = {
                "optimizer": optimizer_search_eval,
                "model": model,
                "train_cache": train_cache,
                "val_loader": val_loader_transformed_preshuffled,
                "transform_seq": transform_seq_arg,
                "dataset_info": dataset_info,
                "architecture": architecture,
                "device": str(device),
                "report_fraction": 0.1,
                "repeats": 1,
            }

            print(f"[{detector}] Running successive halving optimization ({n_trials_halving} trials)...")
            study = run_ood_study_halving(
                study_name=f"{detector}_halving_full_loader",
                storage_path=None,
                detector_name=detector,
                objective_type=search_objective,
                objective_kwargs=objective_kwargs,
                n_trials=n_trials_halving,
                enqueue_params=[copy.deepcopy(default_params)],
            )

            best_params = get_best_ood_params_from_study(study) if study else default_params

            with open(params_path, "w") as f:
                json.dump(best_params, f, indent=2)
            print(f"[{detector}] Saved best parameters to {params_path}")

        # Evaluate if eval_results.json missing or incomplete
        run_evaluation = True
        if os.path.exists(eval_path):
            try:
                with open(eval_path, "r") as f:
                    eval_data = json.load(f)
                if eval_data.get("number_of_runs", 0) >= eval_repeats:
                    print(f"[{detector}] Evaluation already complete, skipping.")
                    run_evaluation = False
            except Exception as e:
                print(f"[{detector}] Warning: Could not read eval JSON ({e}), re-running evaluation.")

        if run_evaluation:
            print(f"[{detector}] Evaluating best config...")
            problem = create_ood_problem(
                detector_name=detector,
                params=best_params,
                model=model,
                train_cache=train_cache,
                transform_seq=transform_seq_arg,
                dataset_info=dataset_info,
                architecture=architecture,
                device=str(device),
            )

            metrics = load_or_run_evaluate_confidence_and_search(
                model=model,
                optimizer=optimizer_search_eval,
                problem=problem,
                test_loader=test_loader_transformed,
                save_path=eval_path,
                max_batch_override=dataset_info.batch_size_search,
                show_progress=show_progress,
                repeats=eval_repeats,
                return_per_run=True,
                overwrite=False,
                store_val=False,
            )

            print(
                f"[{detector}] Final Search Accuracy (Halving): "
                f"{metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}"
            )

    print("\nAll detectors processed successfully (Halving).")

    # %%

    # %%

    # %%
    import pandas as pd

    detectors = ["knn", "knn_mixed", "knn_mixed_faiss", "knn_itf", "vim"]

    # === LOAD RESULTS ===
    def load_metrics(result_dir, detectors):
        results = {}
        for det in detectors:
            eval_path = os.path.join(result_dir, det, "eval_results.json")
            print(f"Loading {eval_path}...")
            if os.path.exists(eval_path):
                with open(eval_path, "r") as f:
                    data = json.load(f)
                    results[det] = {
                        "accuracy_mean": data.get("accuracy_mean", np.nan),
                        "accuracy_se": data.get("accuracy_se", np.nan),
                    }

        return results

    metrics_v1 = load_metrics(
        os.path.join(
            current_path,
            "experiment_files",
            "ood_studies_v1_one_loader",
            str(dataset),
            str(architecture),
            getattr(dataset_info, "transform_seq_name", "default"),
        ),
        detectors, )
    metrics_v2 = load_metrics(
        os.path.join(
            current_path,
            "experiment_files",
            "ood_studies_v2_two_loaders",
            str(dataset),
            str(architecture),
            getattr(dataset_info, "transform_seq_name", "default"),
        ),
        detectors, )
    metrics_v3 = load_metrics(
        os.path.join(
            current_path,
            "experiment_files",
            "ood_studies_v3_three_loaders",
            str(dataset),
            str(architecture),
            getattr(dataset_info, "transform_seq_name", "default"),
        ),
        detectors, )
    metrics_halving = load_metrics(
        os.path.join(
            current_path,
            "experiment_files",
            "ood_studies_halving_full_loader",
            str(dataset),
            str(architecture),
            getattr(dataset_info, "transform_seq_name", "default"),
        ),
        detectors, )

    # === CREATE COMPARISON TABLE ===
    comparison_data = []
    for det in detectors:
        v1 = metrics_v1[det]
        v2 = metrics_v2[det]
        v3 = metrics_v3[det]
        v4 = metrics_halving[det]
        comparison_data.append({
            "Detector": det,
            "V1_Accuracy": v1["accuracy_mean"],
            "V1_se": v1["accuracy_se"],
            "V2_Accuracy": v2["accuracy_mean"],
            "V2_se": v2["accuracy_se"],
            "V3_Accuracy": v3["accuracy_mean"],
            "V3_se": v3["accuracy_se"],
            "V4_Accuracy": v4["accuracy_mean"],
            "V4_se": v4["accuracy_se"],
            "Δ_Accuracy_V3_V1": v3["accuracy_mean"] - v1["accuracy_mean"],
            "Δ_Accuracy_V3_V2": v3["accuracy_mean"] - v2["accuracy_mean"],
            "Δ_Accuracy": v2["accuracy_mean"] - v1["accuracy_mean"],

        })

    df_compare = pd.DataFrame(comparison_data)
    print("\n=== OOD Detector Comparison ===")
    print(df_compare.to_string(index=False))

    # %%

    # %%

    # %%
    # === VISUALIZE RESULTS ===
    x = np.arange(len(detectors))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5 * width, df_compare["V1_Accuracy"], width, label='V1', color='skyblue', alpha=0.7)
    ax.bar(x - 0.5 * width, df_compare["V2_Accuracy"], width, label='V2', color='salmon', alpha=0.7)
    ax.bar(x + 0.5 * width, df_compare["V3_Accuracy"], width, label='V3', color='lightgreen', alpha=0.7)
    # print error bars
    ax.errorbar(x - 1.5 * width, df_compare["V1_Accuracy"], yerr=df_compare["V1_se"], fmt='none', ecolor='blue',
                capsize=5)
    ax.errorbar(x - 0.5 * width, df_compare["V2_Accuracy"], yerr=df_compare["V2_se"], fmt='none', ecolor='red',
                capsize=5)
    ax.errorbar(x + 0.5 * width, df_compare["V3_Accuracy"], yerr=df_compare["V3_se"], fmt='none', ecolor='green',
                capsize=5)
    ax.bar(x + 1.5 * width, df_compare["V4_Accuracy"], width, label='Halving', color='orange', alpha=0.7)
    ax.errorbar(x + 1.5 * width, df_compare["V4_Accuracy"], yerr=df_compare["V4_se"], fmt='none', ecolor='darkorange',
                capsize=5)

    ax.set_ylabel('Accuracy')
    ax.set_title(f'OOD Detector Accuracy Comparison ({dataset}, {architecture})')
    ax.set_xticks(x)
    ax.set_xticklabels(detectors, rotation=25)
    min_accuracy = df_compare[["V1_Accuracy", "V2_Accuracy", "V3_Accuracy"]].min().min()
    max_accuracy = df_compare[["V1_Accuracy", "V2_Accuracy", "V3_Accuracy"]].max().max()
    ax.set_ylim(min_accuracy - 0.05, max_accuracy + 0.05)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # calculate the average accuracy per version
    avg_v1 = df_compare["V1_Accuracy"].mean()
    avg_v2 = df_compare["V2_Accuracy"].mean()
    avg_v3 = df_compare["V3_Accuracy"].mean()
    avg_v4 = df_compare["V4_Accuracy"].mean()
    print(f"Average Accuracy V1: {avg_v1:.4f}")
    print(f"Average Accuracy V2: {avg_v2:.4f}")
    print(f"Average Accuracy V3: {avg_v3:.4f}")
    print(f"Average Accuracy V4: {avg_v4:.4f}")
    # %%
    # === Δ Accuracy Plot ===
    plt.figure(figsize=(8, 4))
    plt.bar(detectors, df_compare["Δ_Accuracy_V3_V1"], color='blue', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel("Δ Accuracy (V3 - V1)")
    plt.title("Performance Improvement per Detector")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    # %% md

    # %%
    import pandas as pd
    for dataset_vis in ["bigger_mnist", "bigger_emnist", "coil100", "tu_berlin", "modelnet10"]:
        dataset2 = dataset_vis
        architecture = default_architecutre_mapping[dataset2]

        experiment_files_path_data = os.path.join(current_path, "experiment_files", "data")
        dataset_info2 = get_dataset_info(dataset2)

        detectors = ["knn", "knn_mixed", "knn_mixed_faiss", "knn_itf", "vim"]

        # === LOAD RESULTS ===
        def load_metrics(result_dir, detectors):
            results = {}
            for det in detectors:
                eval_path = os.path.join(result_dir, det, "eval_results.json")
                print(f"Loading {eval_path}...")
                if os.path.exists(eval_path):
                    with open(eval_path, "r") as f:
                        data = json.load(f)
                        results[det] = {
                            "accuracy_mean": data.get("accuracy_mean", np.nan),
                            "accuracy_std": data.get("accuracy_std", np.nan),
                        }

            return results

        metrics_v1 = load_metrics(
            os.path.join(
                current_path,
                "experiment_files",
                "ood_studies_v1_one_loader",
                str(dataset2),
                str(architecture),
                getattr(dataset_info2, "transform_seq_name", "default"),
            ),
            detectors, )
        metrics_v2 = load_metrics(
            os.path.join(
                current_path,
                "experiment_files",
                "ood_studies_v2_two_loaders",
                str(dataset2),
                str(architecture),
                getattr(dataset_info2, "transform_seq_name", "default"),
            ),
            detectors, )
        metrics_v3 = load_metrics(
            os.path.join(
                current_path,
                "experiment_files",
                "ood_studies_v3_three_loaders",
                str(dataset2),
                str(architecture),
                getattr(dataset_info2, "transform_seq_name", "default"),
            ),
            detectors, )

        metrics_halving = load_metrics(
            os.path.join(
                current_path,
                "experiment_files",
                "ood_studies_halving_full_loader",
                str(dataset2),
                str(architecture),
                getattr(dataset_info2, "transform_seq_name", "default"),
            ),
            detectors, )

        # === CREATE COMPARISON TABLE ===
        comparison_data = []
        for det in detectors:
            v1 = metrics_v1[det]
            v2 = metrics_v2[det]
            v3 = metrics_v3[det]
            v4 = metrics_halving[det]
            comparison_data.append({
                "Detector": det,
                "V1_Accuracy": v1["accuracy_mean"],
                "V1_Std": v1["accuracy_std"],
                "V2_Accuracy": v2["accuracy_mean"],
                "V2_Std": v2["accuracy_std"],
                "V3_Accuracy": v3["accuracy_mean"],
                "V3_Std": v3["accuracy_std"],
                "V4_Accuracy": v4["accuracy_mean"],
                "V4_Std": v4["accuracy_std"],
                "Δ_Accuracy_V3_V1": v3["accuracy_mean"] - v1["accuracy_mean"],
                "Δ_Accuracy_V3_V2": v3["accuracy_mean"] - v2["accuracy_mean"],
                "Δ_Accuracy": v2["accuracy_mean"] - v1["accuracy_mean"],
            })

        df_compare = pd.DataFrame(comparison_data)
        print("\n=== OOD Detector Comparison ===")
        print(df_compare.to_string(index=False))

        # === VISUALIZE RESULTS ===
        x = np.arange(len(detectors))
        width = 0.2

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - 1.5 * width, df_compare["V1_Accuracy"], width, label='V1', color='skyblue', alpha=0.7)
        ax.bar(x - 0.5 * width, df_compare["V2_Accuracy"], width, label='V2', color='salmon', alpha=0.7)
        ax.bar(x + 0.5 * width, df_compare["V3_Accuracy"], width, label='V3', color='lightgreen', alpha=0.7)
        ax.bar(x + 1.5 * width, df_compare["V4_Accuracy"], width, label='Halving', color='orange', alpha=0.7)
        # print error bars

        ax.set_ylabel('Accuracy')
        ax.set_title(f'OOD Detector Accuracy Comparison ({dataset}, {architecture})')
        ax.set_xticks(x)
        ax.set_xticklabels(detectors, rotation=25)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

        # === Δ Accuracy Plot ===
        plt.figure(figsize=(8, 4))
        plt.bar(detectors, df_compare["Δ_Accuracy"], color='green', alpha=0.7)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.ylabel("Δ Accuracy (V2 - V1)")
        plt.title("Performance Improvement per Detector")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        print(f"Finished analysis for dataset: {dataset2}")
        print("Mean V1 Accuracy:", df_compare["V1_Accuracy"].mean())
        print("Mean V2 Accuracy:", df_compare["V2_Accuracy"].mean())
        print("Mean V3 Accuracy:", df_compare["V3_Accuracy"].mean())
        print("Mean V4 Accuracy:", df_compare["V4_Accuracy"].mean())

    # %%
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # List of all datasets to process
    datasets_to_process = ["bigger_mnist", "bigger_emnist", "coil100", "tu_berlin", "modelnet10"]
    detectors_filtered = ["knn", "knn_mixed", "vim"]

    # Store results for all datasets
    all_results = {}

    for dataset_vis in datasets_to_process:
        dataset2 = dataset_vis
        architecture = default_architecutre_mapping[dataset2]
        dataset_info2 = get_dataset_info(dataset2)

        # === LOAD RESULTS ===
        def load_metrics(result_dir, detectors):
            results = {}
            for det in detectors:
                eval_path = os.path.join(result_dir, det, "eval_results.json")
                if os.path.exists(eval_path):
                    try:
                        with open(eval_path, "r") as f:
                            data = json.load(f)
                            results[det] = {
                                "accuracy_mean": data.get("accuracy_mean", np.nan),
                                "accuracy_std": data.get("accuracy_std", np.nan),
                            }
                    except Exception as e:
                        print(f"Error loading {eval_path}: {e}")
                else:
                    print(f"File not found: {eval_path}")
            return results

        metrics_v1 = load_metrics(
            os.path.join(
                current_path,
                "experiment_files",
                "ood_studies_v1_one_loader",
                str(dataset2),
                str(architecture),
                getattr(dataset_info2, "transform_seq_name", "default"),
            ),
            detectors_filtered,
        )
        metrics_v2 = load_metrics(
            os.path.join(
                current_path,
                "experiment_files",
                "ood_studies_v2_two_loaders",
                str(dataset2),
                str(architecture),
                getattr(dataset_info2, "transform_seq_name", "default"),
            ),
            detectors_filtered,
        )
        metrics_v3 = load_metrics(
            os.path.join(
                current_path,
                "experiment_files",
                "ood_studies_v3_three_loaders",
                str(dataset2),
                str(architecture),
                getattr(dataset_info2, "transform_seq_name", "default"),
            ),
            detectors_filtered,
        )
        metrics_halving = load_metrics(
            os.path.join(
                current_path,
                "experiment_files",
                "ood_studies_halving_full_loader",
                str(dataset2),
                str(architecture),
                getattr(dataset_info2, "transform_seq_name", "default"),
            ),
            detectors_filtered,
        )

        # === CREATE COMPARISON TABLE ===
        comparison_data = []
        for det in detectors_filtered:
            v1 = metrics_v1.get(det, {"accuracy_mean": np.nan, "accuracy_std": np.nan})
            v2 = metrics_v2.get(det, {"accuracy_mean": np.nan, "accuracy_std": np.nan})
            v3 = metrics_v3.get(det, {"accuracy_mean": np.nan, "accuracy_std": np.nan})
            v4 = metrics_halving.get(det, {"accuracy_mean": np.nan, "accuracy_std": np.nan})

            comparison_data.append({
                "Detector": det,
                "V1_Accuracy": v1["accuracy_mean"],
                "V1_Std": v1["accuracy_std"],
                "V2_Accuracy": v2["accuracy_mean"],
                "V2_Std": v2["accuracy_std"],
                "V3_Accuracy": v3["accuracy_mean"],
                "V3_Std": v3["accuracy_std"],
                "V4_Accuracy": v4["accuracy_mean"],
                "V4_Std": v4["accuracy_std"],
            })

        df_compare = pd.DataFrame(comparison_data)
        all_results[dataset_vis] = df_compare

        print(f"\n=== OOD Detector Comparison ({dataset_vis}) ===")
        print(df_compare.to_string(index=False))

    # === GENERATE PLOTS FOR ALL DATASETS ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, dataset_vis in enumerate(datasets_to_process):
        df_compare = all_results[dataset_vis]
        architecture = default_architecutre_mapping[dataset_vis]

        x = np.arange(len(detectors_filtered))
        width = 0.2

        ax = axes[idx]
        ax.bar(x - 1.5 * width, df_compare["V1_Accuracy"], width, label='V1', color='skyblue', alpha=0.7)
        ax.bar(x - 0.5 * width, df_compare["V2_Accuracy"], width, label='V2', color='salmon', alpha=0.7)
        ax.bar(x + 0.5 * width, df_compare["V3_Accuracy"], width, label='V3', color='lightgreen', alpha=0.7)
        ax.bar(x + 1.5 * width, df_compare["V4_Accuracy"], width, label='Halving', color='orange', alpha=0.7)

        ax.set_ylabel('Accuracy')
        ax.set_title(f'{dataset_vis} ({architecture})')
        ax.set_xticks(x)
        ax.set_xticklabels(detectors_filtered, rotation=0)
        ax.legend(fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(0, 1)

    # Hide the extra subplot
    axes[-1].axis('off')

    plt.suptitle('OOD Detector Accuracy Comparison (All Datasets)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # === PRINT SUMMARY STATISTICS ===
    print("\n=== SUMMARY STATISTICS ===")
    for dataset_vis in datasets_to_process:
        df_compare = all_results[dataset_vis]
        print(f"\n{dataset_vis.upper()}:")
        print(f"  Mean V1 Accuracy: {df_compare['V1_Accuracy'].mean():.4f}")
        print(f"  Mean V2 Accuracy: {df_compare['V2_Accuracy'].mean():.4f}")
        print(f"  Mean V3 Accuracy: {df_compare['V3_Accuracy'].mean():.4f}")
        print(f"  Mean V4 Accuracy: {df_compare['V4_Accuracy'].mean():.4f}")
    # %%
    # === GENERATE PLOTS FOR ALL DATASETS WITH ERROR BARS ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, dataset_vis in enumerate(datasets_to_process):
        df_compare = all_results[dataset_vis]
        architecture = default_architecutre_mapping[dataset_vis]

        x = np.arange(len(detectors_filtered))
        width = 0.2

        ax = axes[idx]
        # Bars with error bars
        ax.bar(x - 1.5 * width, df_compare["V1_Accuracy"], width, label='V1', color='skyblue', alpha=0.7,
               yerr=df_compare["V1_Std"], capsize=5, error_kw={'elinewidth': 1, 'ecolor': 'blue'})
        ax.bar(x - 0.5 * width, df_compare["V2_Accuracy"], width, label='V2', color='salmon', alpha=0.7,
               yerr=df_compare["V2_Std"], capsize=5, error_kw={'elinewidth': 1, 'ecolor': 'red'})
        ax.bar(x + 0.5 * width, df_compare["V3_Accuracy"], width, label='V3', color='lightgreen', alpha=0.7,
               yerr=df_compare["V3_Std"], capsize=5, error_kw={'elinewidth': 1, 'ecolor': 'green'})
        ax.bar(x + 1.5 * width, df_compare["V4_Accuracy"], width, label='Halving', color='orange', alpha=0.7,
               yerr=df_compare["V4_Std"], capsize=5, error_kw={'elinewidth': 1, 'ecolor': 'darkorange'})

        ax.set_ylabel('Accuracy')
        ax.set_title(f'{dataset_vis} ({architecture})')
        ax.set_xticks(x)
        ax.set_xticklabels(detectors_filtered, rotation=0)
        ax.legend(fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(0, 1)

    # Hide the extra subplot if there is one
    if len(datasets_to_process) < len(axes):
        for extra_ax in axes[len(datasets_to_process):]:
            extra_ax.axis('off')

    plt.suptitle('OOD Detector Accuracy Comparison (All Datasets)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # %%
    # === CALCULATE MEANS PER DATASET WITH DESCRIPTIVE NAMES ===
    summary_data = []

    method_names = ["1 Val Set", "2 Val Sets", "3 Val Sets", "Halving"]

    for dataset_vis in datasets_to_process:
        df_compare = all_results[dataset_vis]
        summary_data.append({
            "Dataset": dataset_vis,
            "1 Val Set": df_compare["V1_Accuracy"].mean(),
            "2 Val Sets": df_compare["V2_Accuracy"].mean(),
            "3 Val Sets": df_compare["V3_Accuracy"].mean(),
            "Halving": df_compare["V4_Accuracy"].mean(),
        })

    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_data)

    # === CALCULATE OVERALL MEAN ACROSS ALL DATASETS ===
    overall_means = summary_df[method_names].mean()
    overall_row = {"Dataset": "Overall"}
    overall_row.update(overall_means.to_dict())
    summary_df = pd.concat([summary_df, pd.DataFrame([overall_row])], ignore_index=True)

    # === MARK THE BEST VALUE PER ROW WITH LaTeX BOLD ===
    def highlight_best(row):
        row_copy = row.copy()
        # Skip the Dataset column
        values = row[method_names]
        max_val = values.max()
        for method in method_names:
            if values[method] == max_val:
                row_copy[method] = f"\\textbf{{{values[method]:.4f}}}"
            else:
                row_copy[method] = f"{values[method]:.4f}"
        # Keep the Dataset column unchanged
        row_copy["Dataset"] = row["Dataset"]
        return row_copy

    highlighted_df = summary_df.apply(highlight_best, axis=1)

    # Export to LaTeX
    latex_table = highlighted_df.to_latex(index=False, escape=False,
                                          caption="Mean Accuracy per Dataset with Different Validation Strategies (Best in Bold)",
                                          label="tab:mean_accuracy")
    print(latex_table)

    # %%

def run_3_1():
    # %%
    import copy

    import torch
    import torch.nn as nn
    import torchvision
    import numpy as np
    from matplotlib import pyplot as plt

    from its.search import InverseTransformationSearch
    from search.parallel_gradient import ParallelGradientDescent
    from utils.affine_transforms_old import AffineTransformation2D
    from utils.sampling import BatchNegativeSampler

    # torch.cuda.is_available = lambda: False
    # device = torch.device("cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # look for experiment files in parents
    import os

    path_found = False
    current_path = os.getcwd()
    while not path_found:
        if os.path.exists(os.path.join(current_path, "experiment_files")):
            path_found = True
            break
        current_path = os.path.dirname(current_path)

    experiment_files_path_data = os.path.join(current_path, "experiment_files", "data")
    dataset = "modelnet10"

    default_architecutre_mapping = {
        "mnist": "resnet_small",
        "bigger_mnist": "resnet_small",
        "emnist": "extended_resnet_small",
        "bigger_emnist": "bigger_extended_resnet_small",
        "coil100": "coil_resnet_small",
        "tu_berlin": "bi_lstm",
        "modelnet10": "pointnetplus",
    }

    architecture = default_architecutre_mapping[dataset]
    budget = None
    # %%
    # TODO prototype multi has to be rerun
    # %%
    from experiment_thesis.dataset_preperation.get_dataset import get_dataset_info, get_dataset

    dataset_info = get_dataset_info(dataset)

    dataset_dict = get_dataset(dataset_info, path=experiment_files_path_data, batch_size=dataset_info.batch_size)
    transform_name = dataset_info.transform_seq_name
    # %%

    dataset_dict.keys()
    dataset_train = dataset_dict['train_dataset']
    dataset_val = dataset_dict['val_dataset']
    dataset_test = dataset_dict['test_dataset']
    train_loader = dataset_dict['train_loader']
    val_loader = dataset_dict['val_loader']
    test_loader = dataset_dict['test_loader']
    n_classes = dataset_info.num_classes
    train_loader_transformed = dataset_dict['train_loader_transformed']
    val_loader_transformed = dataset_dict['val_loader_transformed']
    test_loader_transformed = dataset_dict['test_loader_transformed']
    train_loader_no_shuffle = dataset_dict['train_loader_no_shuffle']
    # %%
    x = next(iter(test_loader_transformed))[0]

    batch_size = next(iter(train_loader))[0].shape[0]

    from utils.eval.vis import vis_dataset

    vis_dataset(train_loader, val_loader, test_loader_transformed)
    from experiment_thesis.main import train_and_get_model, train_or_load_energy_model
    from experiment_thesis.dataset_preperation.basic_networks import get_network
    from utils.eval.main_model import evaluate_base_model

    model_dir_path = os.path.join(current_path, "experiment_files", "models")
    embedding_cache_path = os.path.join(current_path, "experiment_files", "embedding_cache")
    # Add results dir and helper for save paths
    results_dir_path = os.path.join(current_path, "experiment_files", "results", dataset, architecture,
                                    "comparison_unsupervised")
    os.makedirs(results_dir_path, exist_ok=True)

    def savepath(label: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
        return os.path.join(results_dir_path, transform_name, f"{safe}.json")

    # %%
    model = get_network(dataset_info, architecture, num_classes=n_classes).to(device)
    modelname = f"{dataset}_{architecture}"
    cache_name_train = f"{dataset}_{architecture}_embedding_cache_train"

    train_and_get_model(model, model_dir_path, modelname, train_loader, val_loader, trainer_kwargs={
        "accelerator": "auto",
        "max_epochs": dataset_info.epochs,
        "precision": "16-mixed",
    }, load_if_exists=True)

    # %%
    model.eval().to(device)
    # %%
    # check main model
    res = evaluate_base_model(model, test_loader_transformed, device)
    print(res)
    res = evaluate_base_model(model, test_loader, device)
    print(res)
    # %%

    # %%
    is_image_data = len(dataset_info.input_size) == 3 and dataset_info.input_size[0] in [1, 3]

    from utils.transforms.apply import grid_resample
    from experiment_thesis.dataset_preperation.transformation import get_transformation_sequence_images

    transform_seq = get_transformation_sequence_images(
        name=dataset_info.transform_seq_name,
        resample_method=dataset_info.resample_method,
        init_method="sobol"
    ).to(device)
    from utils.replacer import replace_rotation_transforms_2vec

    if dataset == "modelnet10":
        transform_seq = replace_rotation_transforms_2vec(transform_seq)
    # %%
    print(transform_seq.transformations)
    # %%
    from experiment_thesis.dataset_preperation.basic_networks import make_deterministic
    make_deterministic(model)

    # %%
    from experiment_thesis.dataset_preperation.get_dataset import get_layer_embedding_cache_config, \
        create_layer_embedding_cache
    cache_config = get_layer_embedding_cache_config(dataset, architecture, transform_name=None,
                                                    dataset_info=dataset_info)
    train_cache = create_layer_embedding_cache(model, train_loader_no_shuffle, cache_config, embedding_cache_path,
                                               device=device)

    # %%
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # %%
    from experiment_thesis.ood import base_prepare

    base_prepare.OOD_PARAM_SAMPLERS.keys()
    # %%
    detectors = ["knn", "per_class_knn", "knn_mixed", "per_class_knn_mixed", "knn_mixed_faiss", "knn_itf", "vim",
                 "react", "dice", "ash", "she", "laplace_mi", "laplace_energy", "laplace_weighted", "trust_score",
                 "openmax", "mahalanobis", "rmd", "class_prototype", "react_all", "energy", "per_class_prototype",
                 "single_mahalanobis", "single_rmd", "single_mahalanobis_individual", "single_rmd_individual",
                 "mahalanobis_individual", "rmd_individual", "prototype_multi", "laplace_entropy", "adjusted_entropy",
                 "nn_guided", "nn_guided_one"]
    # dont do kde to expensive, and not better.
    # global protoype makes no sense over class based protoype
    # lof fit is to slow, gmm is even slower or fails due to size
    # gram runs out of main memory,

    detectors_parameterless = ["energy_ts", "entropy", "kl_matching", "laplace_entropy_gridsearch"]
    detectors = detectors + detectors_parameterless
    from experiment_thesis.dataset_preperation.basic_networks import FlexibleResNet
    if not isinstance(model, FlexibleResNet):
        # remove ash and react_all as they only work with resnets
        detectors.remove("ash")
        detectors.remove("react_all")
        if "react" not in detectors:
            detectors.append("react")
        detectors.append("ash_last")

    # %%

    # %%

    # %%

    # %%
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # %%

    # %%
    import os
    import json
    import torch
    import gc
    import numpy as np
    import copy
    from torch.utils.data import DataLoader, Subset
    import optuna

    from experiment_thesis.ood.base_prepare import (
        run_ood_study,
        get_default_ood_params,
        get_best_ood_params_from_study,
        create_ood_problem,
    )
    from utils.eval.ood_performance import load_or_run_evaluate_confidence_and_search
    from search.shgo import SHGO

    search_objective = "search"

    n_trials_search = 120
    n_trials_search_refine = 15
    eval_repeats = 5
    if dataset in ["tu_berlin", "modelnet10", "coil100"]:
        eval_repeats = 10
    show_progress = True
    top_k = 3

    # this here forces the main experiment to not load models from smaller loaders
    # as this can lead to overfitting to small loaders which is not desired.
    # only the loader on stage 2 the full val set should be stored and reused for final eval(so that they are the same as the best selected model)
    allow_loading_models_from_smaller_loaders = False

    rng = np.random.default_rng(seed=42)

    # Determine report_fraction based on dataset
    if dataset in ["mnist", "bigger_mnist", "emnist", "bigger_emnist"]:
        report_fraction_stage1 = 0.25
    else:
        report_fraction_stage1 = 0.5

    transform_seq_arg = transform_seq

    optimizer = SHGO(initial_samples=46, local_runs=2, local_max_steps=3, local_opt_kwargs={"lr": 0.1})
    if dataset == "tu_berlin":
        optimizer = SHGO(initial_samples=60, local_runs=1, local_max_steps=0, local_opt_kwargs={"lr": 0.1})

    optimizer_search_eval = optimizer

    base_results_dir = os.path.join(
        current_path,
        "experiment_files",
        "results",
        "comparison_unsupervised",
        str(dataset),
        str(architecture),
        getattr(dataset_info, "transform_seq_name", "default"),
    )
    os.makedirs(base_results_dir, exist_ok=True)

    model.eval().to(device)

    # ---------------- Prepare validation subsets ----------------
    val_dataset = val_loader_transformed.dataset
    n_samples = len(val_dataset)
    subset_size = n_samples // 6

    all_indices = rng.permutation(n_samples)

    subset_indices_1 = all_indices[:subset_size]
    subset_indices_2 = all_indices[subset_size: 2 * subset_size]
    subset_indices_3 = all_indices[2 * subset_size: 3 * subset_size]

    val_subset_1 = Subset(val_dataset, subset_indices_1)
    val_subset_2 = Subset(val_dataset, subset_indices_2)
    val_subset_3 = Subset(val_dataset, subset_indices_3)

    val_loader_small_1 = DataLoader(
        val_subset_1,
        batch_size=dataset_info.batch_size,
        shuffle=False,
        num_workers=val_loader_transformed.num_workers,
        pin_memory=True,
        persistent_workers=val_loader_transformed.persistent_workers
    )
    val_loader_small_2 = DataLoader(
        val_subset_2,
        batch_size=dataset_info.batch_size,
        shuffle=False,
        num_workers=val_loader_transformed.num_workers,
        pin_memory=True,
        persistent_workers=val_loader_transformed.persistent_workers
    )
    val_loader_small_3 = DataLoader(
        val_subset_3,
        batch_size=dataset_info.batch_size,
        shuffle=False,
        num_workers=val_loader_transformed.num_workers,
        pin_memory=True,
        persistent_workers=val_loader_transformed.persistent_workers
    )

    val_loader_transformed_preshuffled = DataLoader(
        Subset(val_dataset, all_indices),
        batch_size=dataset_info.batch_size,
        shuffle=False,
        num_workers=val_loader_transformed.num_workers,
        pin_memory=True,
        persistent_workers=val_loader_transformed.persistent_workers
    )

    val_dataset_in = val_loader.dataset

    val_loader_small_in_dist = DataLoader(
        Subset(val_dataset_in, subset_indices_1),
        batch_size=dataset_info.batch_size,
        shuffle=False,
        num_workers=val_loader_transformed.num_workers,
        pin_memory=True,
        persistent_workers=val_loader_transformed.persistent_workers
    )
    val_loader_small_in_dist2 = DataLoader(
        Subset(val_dataset_in, subset_indices_2),
        batch_size=dataset_info.batch_size,
        shuffle=False,
        num_workers=val_loader_transformed.num_workers,
        pin_memory=True,
        persistent_workers=val_loader_transformed.persistent_workers
    )

    val_loader_small_in_dist3 = DataLoader(
        Subset(val_dataset_in, subset_indices_3),
        batch_size=dataset_info.batch_size,
        shuffle=False,
        num_workers=val_loader_transformed.num_workers,
        pin_memory=True,
        persistent_workers=val_loader_transformed.persistent_workers
    )

    val_loader_preshuffled_in_dist = DataLoader(
        Subset(val_dataset_in, all_indices),
        batch_size=dataset_info.batch_size,
        shuffle=False,
        num_workers=val_loader_transformed.num_workers,
        pin_memory=True,
        persistent_workers=val_loader_transformed.persistent_workers
    )

    # enable grad for model %TODO remove
    for param in model.parameters():
        param.requires_grad = False

    print(f"Each subset: {subset_size} samples ({subset_size / n_samples:.1%} of dataset)")
    print("Two disjoint subsets created.\n")

    # ---------------- Loop over detectors ----------------
    for detector in detectors:
        print(f"\n=== Detector: {detector} ===")
        detector_dir = os.path.join(base_results_dir, detector)
        os.makedirs(detector_dir, exist_ok=True)

        params_path = os.path.join(detector_dir, "best_params.json")
        best_model_path_prefix = os.path.join(detector_dir, "best_model")  # Prefix for model files
        eval_path = os.path.join(detector_dir, "eval_results.json")
        eval_path_default = os.path.join(detector_dir, "eval_results_default.json")
        stage1_params_path = os.path.join(detector_dir, "stage1_best_params.json")

        if os.path.exists(params_path) and os.path.exists(eval_path) and os.path.exists(eval_path_default):
            print(f"[{detector}] Found cached results, skipping.")
            continue

        gc.collect();
        torch.cuda.empty_cache()
        default_params = get_default_ood_params(detector)

        if detector not in detectors_parameterless:
            print(f"[{detector}] Starting two-stage hyperparameter optimization...")
            # Load stage1 params if exist
            best_stage1_candidates = []  # Now list of (score, params, model_paths)
            val_loaders_small = [val_loader_small_1, val_loader_small_2, val_loader_small_3]
            trials_per_loader = [n_trials_search // 3, n_trials_search // 3,
                                 n_trials_search - n_trials_search // 3 - n_trials_search // 3]
            val_loadders_small_in_dist = [val_loader_small_in_dist, val_loader_small_in_dist2,
                                          val_loader_small_in_dist3]

            for i, (small_loader, n_trials_this_run) in enumerate(zip(val_loaders_small, trials_per_loader), start=1):
                part_params_path = os.path.join(detector_dir, f"stage1_part{i}_best_params.json")
                part_scores_path = os.path.join(detector_dir, f"stage1_part{i}_scores.json")
                part_model_paths_path = os.path.join(detector_dir,
                                                     f"stage1_part{i}_model_paths.json")  # New file for model paths

                if os.path.exists(part_params_path) and os.path.exists(part_scores_path):
                    with open(part_params_path, "r") as f_p, open(part_scores_path, "r") as f_s:
                        part_params = json.load(f_p)
                        part_scores = json.load(f_s)
                    part_model_paths = []
                    if os.path.exists(part_model_paths_path):
                        with open(part_model_paths_path, "r") as f:
                            part_model_paths = json.load(f)
                    else:
                        part_model_paths = [None] * len(part_params)
                    best_stage1_candidates.extend(zip(part_scores, part_params, part_model_paths))
                    print(f"[{detector}] Loaded {len(part_params)} stage1 part {i} candidates from {part_params_path}.")
                    continue

                print(f"\n[{detector}] Running on coarse loader {i}/3 ({n_trials_this_run} trials)...")

                objective_kwargs_stage1 = {
                    "optimizer": optimizer,
                    "model": model,
                    "train_cache": train_cache,
                    "val_loader": small_loader,
                    "transform_seq": transform_seq_arg,
                    "dataset_info": dataset_info,
                    "architecture": architecture,
                    "device": str(device),
                    "report_fraction": report_fraction_stage1,
                    "repeats": 1,
                    "val_id_loader": val_loadders_small_in_dist[i - 1],
                    "val_ood_loader": small_loader,
                }

                study_stage1 = run_ood_study(
                    study_name=f"{detector}_stage1_part{i}",
                    storage_path=None,
                    detector_name=detector,
                    objective_type=search_objective,
                    objective_kwargs=objective_kwargs_stage1,
                    n_trials=n_trials_this_run,
                    enqueue_params=[copy.deepcopy(default_params)],
                )

                part_params = []
                part_scores = []
                part_model_paths = []  # New list for model paths
                if study_stage1 is not None:
                    # --- FILTER ONLY COMPLETE TRIALS ---
                    completed_trials = [
                        t for t in study_stage1.trials
                        if t.state == optuna.trial.TrialState.COMPLETE
                    ]
                    if not completed_trials:
                        print(f"[{detector}] Warning: No completed trials in subset {i}")
                    else:
                        topk_trials = sorted(
                            completed_trials,
                            key=lambda t: t.value,  # only final value
                            reverse=True
                        )[:top_k]

                        print(f"[{detector}] Top-{top_k} parameters from subset {i}:")
                        for rank, t in enumerate(topk_trials, start=1):
                            t_copy = copy.deepcopy(t.params)
                            part_params.append(t_copy)
                            part_scores.append(t.value)
                            print(f"  Rank {rank}: value={t.value:.4f}, params={t_copy}")

                            # --- Save model params if they exist ---
                            if "model_params" in t.user_attrs:
                                model_params = t.user_attrs["model_params"]
                                model_paths = []
                                for model_idx, model_state in enumerate(model_params):
                                    model_path = os.path.join(detector_dir,
                                                              f"stage1_part{i}_trial_{t.number}_model_{model_idx}.pt")
                                    torch.save(model_state, model_path)
                                    model_paths.append(model_path)
                                part_model_paths.append(model_paths)
                                print(f"    -> Saved {len(model_paths)} model(s).")
                            else:
                                part_model_paths.append(None)

                # Save part params, scores, and model paths
                with open(part_params_path, "w") as f:
                    json.dump(part_params, f, indent=2)
                with open(part_scores_path, "w") as f:
                    json.dump(part_scores, f, indent=2)
                with open(part_model_paths_path, "w") as f:
                    json.dump(part_model_paths, f, indent=2)
                print(f"[{detector}] Saved {len(part_params)} stage1 part {i} best params and scores.")
                best_stage1_candidates.extend(zip(part_scores, part_params, part_model_paths))

            print(f"[{detector}] Stage1 completed, total candidates: {len(best_stage1_candidates)}.")

            # === Stage 2 ===
            if os.path.exists(params_path):
                with open(params_path, "r") as f:
                    best_params = json.load(f)
                print(f"[{detector}] Loaded best params from {params_path}, skipping stage2.")
            else:
                print(f"\n[{detector}] Stage 2: refine search...")
                objective_kwargs_stage2 = {
                    "optimizer": optimizer,
                    "model": model,
                    "train_cache": train_cache,
                    "val_loader": val_loader_transformed_preshuffled,
                    "transform_seq": transform_seq_arg,
                    "dataset_info": dataset_info,
                    "architecture": architecture,
                    "device": str(device),
                    "report_fraction": 0.1,
                    "repeats": 1,
                    "val_id_loader": val_loader_preshuffled_in_dist,
                    "val_ood_loader": val_loader_transformed_preshuffled,
                }

                # --- Sort candidates by score (descending) ---
                best_stage1_candidates.sort(key=lambda x: x[0], reverse=True)
                sorted_best_params = [p for s, p, m in best_stage1_candidates]
                sorted_model_paths = [m for s, p, m in best_stage1_candidates]

                # --- Enqueue best-first, loading models ---
                enqueue_list = []
                # Add default params first
                enqueue_list.append((copy.deepcopy(default_params), {}))

                # Add sorted candidates with their model states
                for params, model_paths in zip(sorted_best_params, sorted_model_paths):
                    user_attrs = {}
                    if model_paths and allow_loading_models_from_smaller_loaders:
                        loaded_model_params = [torch.load(mp, map_location="cpu") for mp in model_paths if
                                               os.path.exists(mp)]
                        if loaded_model_params:
                            user_attrs["model_params"] = loaded_model_params
                    enqueue_list.append((params, user_attrs))

                print(f"[{detector}] Enqueuing {len(enqueue_list)} parameter sets for Stage 2 refinement (best first):")
                for idx, (p, ua) in enumerate(enqueue_list, start=1):
                    print(f"  Enqueued {idx}: {p}" + (" (with model)" if ua else ""))

                study_stage2 = run_ood_study(
                    study_name=f"{detector}_stage2",
                    storage_path=None,
                    detector_name=detector,
                    objective_type=search_objective,
                    objective_kwargs=objective_kwargs_stage2,
                    n_trials=n_trials_search_refine,
                    enqueue_params=enqueue_list,
                )

                best_params = (
                    get_best_ood_params_from_study(study_stage2)
                    if study_stage2
                    else (sorted_best_params[0] if sorted_best_params else default_params)
                )

                # Save best model states if available
                if study_stage2 and study_stage2.best_trial and "model_params" in study_stage2.best_trial.user_attrs:
                    for model_idx, model_state in enumerate(study_stage2.best_trial.user_attrs["model_params"]):
                        torch.save(model_state, f"{best_model_path_prefix}_{model_idx}.pt")
                    print(f"[{detector}] Saved best model states.")

                with open(params_path, "w") as f:
                    json.dump(best_params, f, indent=2)
                print(f"[{detector}] Saved best parameters to {params_path}")

            # Final evaluations
            if not os.path.exists(eval_path):
                print(f"[{detector}] Evaluating final configuration with optimized params...")

                final_kwargs = {
                    "model": model,
                    "train_cache": train_cache,
                    "transform_seq": transform_seq_arg,
                    "dataset_info": dataset_info,
                    "architecture": architecture,
                    "device": str(device),
                    "val_id_loader": val_loader_preshuffled_in_dist,  # Provide loaders for fitting default params
                    "val_ood_loader": val_loader_transformed_preshuffled,
                }

                # Load saved model states for optimized params
                loaded_model_params = []
                i = 0
                while os.path.exists(f"{best_model_path_prefix}_{i}.pt"):
                    loaded_model_params.append(torch.load(f"{best_model_path_prefix}_{i}.pt"))
                    i += 1
                if len(loaded_model_params) > 0:
                    final_kwargs["model_params"] = loaded_model_params
                    print(f"Loaded {len(loaded_model_params)} final model states for evaluation.")

                problem = create_ood_problem(
                    detector_name=detector,
                    params=best_params,
                    **final_kwargs,
                )

                metrics = load_or_run_evaluate_confidence_and_search(
                    model=model,
                    optimizer=optimizer_search_eval,
                    problem=problem,
                    test_loader=test_loader_transformed,
                    save_path=eval_path,
                    max_batch_override=dataset_info.batch_size_search,
                    show_progress=show_progress,
                    repeats=eval_repeats,
                    return_per_run=True,
                    overwrite=False,
                    store_val=False,
                    store_correct=True,
                )

                print(
                    f"[{detector}] Final Search Accuracy (Optimized): "
                    f"{metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}"
                )

        if not os.path.exists(eval_path_default):
            print(f"[{detector}] Evaluating final configuration with default params...")

            final_kwargs_default = {
                "model": model,
                "train_cache": train_cache,
                "transform_seq": transform_seq_arg,
                "dataset_info": dataset_info,
                "architecture": architecture,
                "device": str(device),
                "val_id_loader": val_loader_preshuffled_in_dist,  # Provide loaders for fitting default params
                "val_ood_loader": val_loader_transformed_preshuffled,
            }

            problem_default = create_ood_problem(
                detector_name=detector,
                params=default_params,
                **final_kwargs_default,
            )

            metrics_default = load_or_run_evaluate_confidence_and_search(
                model=model,
                optimizer=optimizer_search_eval,
                problem=problem_default,
                test_loader=test_loader_transformed,
                save_path=eval_path_default,
                max_batch_override=dataset_info.batch_size_search,
                show_progress=show_progress,
                repeats=eval_repeats,
                return_per_run=True,
                overwrite=False,
                store_val=False,
                store_correct=True,
            )

            print(
                f"[{detector}] Final Search Accuracy (Default): "
                f"{metrics_default['accuracy_mean']:.4f} ± {metrics_default['accuracy_std']:.4f}"
            )

    print("\nAll detectors processed successfully.")
    # %%
    # iterate over the detectors and print default and best parameters
    for detector in ["rmd_individual"]:
        detector_dir = os.path.join(base_results_dir, detector)
        params_path = os.path.join(detector_dir, "best_params.json")
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                best_params = json.load(f)
            print(f"\n=== Detector: {detector} ===")
            print("Best Parameters:")
            for k, v in best_params.items():
                print(f"  {k}: {v}")
        else:
            print(f"\n=== Detector: {detector} ===")
            print("No optimized parameters found.")
        # print default parameters
        default_params = get_default_ood_params(detector)
        print("Default Parameters:")
        for k, v in default_params.items():
            print(f"  {k}: {v}")
    # %%

    # %%
    import os
    import json
    import numpy as np
    import pandas as pd

    # === LOAD RESULTS ===
    def load_metrics(result_dir, detectors):
        results = {}
        for det in detectors:
            eval_path = os.path.join(result_dir, det, "eval_results.json")
            eval_path_default = os.path.join(result_dir, det, "eval_results_default.json")
            print(f"Loading {eval_path} and {eval_path_default}...")

            # Initialize metrics with NaN
            metrics_data = {
                "accuracy_mean_optimized": np.nan,
                "accuracy_std_optimized": np.nan,
                "accuracy_mean_default": np.nan,
                "accuracy_std_default": np.nan,
                "accuracy_se_optimized": np.nan,
                "accuracy_se_default": np.nan,
            }

            # Load optimized results if file exists
            if os.path.exists(eval_path):
                with open(eval_path, "r") as f:
                    data = json.load(f)
                metrics_data["accuracy_mean_optimized"] = data.get("accuracy_mean", np.nan)
                metrics_data["accuracy_std_optimized"] = data.get("accuracy_std", np.nan)
                metrics_data["accuracy_se_optimized"] = data.get("accuracy_se", np.nan)

            # Load default results if file exists
            if os.path.exists(eval_path_default):
                with open(eval_path_default, "r") as f:
                    data_default = json.load(f)
                metrics_data["accuracy_mean_default"] = data_default.get("accuracy_mean", np.nan)
                metrics_data["accuracy_std_default"] = data_default.get("accuracy_std", np.nan)
                metrics_data["accuracy_se_default"] = data_default.get("accuracy_se", np.nan)

            results[det] = metrics_data

        return results

    # Example usage
    metrics = load_metrics(base_results_dir, detectors)

    # === CREATE COMPARISON TABLE ===
    comparison_data = []
    for det in detectors:
        m = metrics[det]
        comparison_data.append({
            "Detector": det,
            "Optimized_Accuracy": m["accuracy_mean_optimized"],
            "Optimized_Std": m["accuracy_std_optimized"],
            "Default_Accuracy": m["accuracy_mean_default"],
            "Default_Std": m["accuracy_std_default"],
            "Optimized_SE": m["accuracy_se_optimized"],
            "Default_SE": m["accuracy_se_default"],
        })

    df_compare = pd.DataFrame(comparison_data)
    print("\n=== OOD Detector Comparison ===")
    print(df_compare.to_string(index=False))

    # %% md

    # %%
    import matplotlib.pyplot as plt
    import numpy as np

    # Assuming df_compare is already defined
    df_plot = df_compare.assign(
        Max_Accuracy=df_compare[["Default_Accuracy", "Optimized_Accuracy"]].max(axis=1)
    ).sort_values("Max_Accuracy", ascending=False)

    x = np.arange(len(df_plot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(
        x - width / 2,
        df_plot["Default_Accuracy"],
        width,
        yerr=df_plot["Default_Std"],
        label="Default",
        capsize=5,
    )

    bars2 = ax.bar(
        x + width / 2,
        df_plot["Optimized_Accuracy"],
        width,
        yerr=df_plot["Optimized_Std"],
        label="Optimized",
        capsize=5,
    )

    ax.set_xlabel("Detector", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("OOD Detector Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["Detector"], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # === Zoom in Y-axis to focus on differences ===
    y_min = min(df_plot[["Default_Accuracy", "Optimized_Accuracy"]].min().min(),
                (df_plot[["Default_Accuracy", "Optimized_Accuracy"]].min().min() -
                 df_plot[["Default_Std", "Optimized_Std"]].max().max()))
    y_max = max(df_plot[["Default_Accuracy", "Optimized_Accuracy"]].max().max(),
                (df_plot[["Default_Accuracy", "Optimized_Accuracy"]].max().max() +
                 df_plot[["Default_Std", "Optimized_Std"]].max().max()))

    # Add a small margin for visual breathing room
    margin = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - margin, y_max + margin)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            # ax.text(bar.get_x() + bar.get_width()/2, height + 0.002, f"{height:.3f}",
            #        ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()

    # %%
    from utils.eval.vis import plt_setup_latex

    W = plt_setup_latex()
    # %%
    name_map = {
        "single_rmd": "1L-RMD",
        "single_mahalanobis": "1L-MD",
        "rmd": "RMD",
        "mahalanobis": "MD",
        "knn_mixed_faiss": "KNN Mixed (Faiss)",
        "knn_mixed": "KNN Mixed",
        "nn_guided_one": "NN-Guided (1)",

        "knn": "KNN",
        "per_class_knn_mixed": "Per-Class KNN Mixed",
        "per_class_knn": "Per-Class KNN",
        "knn_itf": "KNN-ITF",

        "trust_score": "Trust Score",
        "vim": "VIM",

        "per_class_prototype": "Per-Class Prototype",
        "prototype_multi": "Prototype (multi)",
        "class_prototype": "Prototype",

        "kl_matching": "KL-Matching",

        "adjusted_entropy": "Adj. Entropy",
        "laplace_entropy": "Lapl. Entropy",
        "laplace_entropy_gridsearch": "Lapl. Entropy (GS)",
        "react_all": "ReAct (All)",
        "entropy": "Entropy",
        "laplace_weighted": "Lapl. Weighted",

        "react": "ReAct",
        "dice": "DICE",
        "laplace_mi": "Lapl. MI",

        # **Your specified preferred naming**
        "nn_guided": "NN-Guidance",
        "ash": "ASH",
        "openmax": "Openmax",
        "energy_ts": "Energy T. Scale",
        "energy": "Energy",
        "laplace_energy": "Lapl. Energy",
        "mahalanobis_individual": "MD-Diag",
        "single_rmd_individual": "1L-RMD-Diag",
        "single_mahalanobis_individual": "1L-MD-Diag",
        "she": "SHE",
        "rmd_individual": "RMD-Diag",
    }

    # %%
    figure_path = os.path.join(current_path, "experiment_files", "export", "fig", "comparision_unsupervised", dataset,
                               transform_name)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    # %%
    import matplotlib.pyplot as plt
    import numpy as np

    # Prepare and rename detectors
    df_plot = df_compare.assign(
        Max_Accuracy=df_compare[["Default_Accuracy", "Optimized_Accuracy"]].max(axis=1)
    ).sort_values("Max_Accuracy", ascending=True)  # ascending → best at top

    df_plot["Detector"] = df_plot["Detector"].replace(name_map)

    y = np.arange(len(df_plot))
    height = 0.3

    fig, ax = plt.subplots(figsize=(W, W * 1.5))

    bars1 = ax.barh(
        y - height / 2,
        df_plot["Default_Accuracy"],
        height,
        xerr=df_plot["Default_SE"],
        label="Default",
        capsize=1,
    )

    bars2 = ax.barh(
        y + height / 2,
        df_plot["Optimized_Accuracy"],
        height,
        xerr=df_plot["Optimized_SE"],
        label="Optimized",
        capsize=1,
    )

    ax.set_ylabel("Detector", fontsize=12)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("OOD Detector Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(df_plot["Detector"])
    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    # === Zoom in X-axis to focus on differences ===
    x_min = min(df_plot["Default_Accuracy"].min(), df_plot["Optimized_Accuracy"].min())
    x_max = max(df_plot["Default_Accuracy"].max(), df_plot["Optimized_Accuracy"].max())
    margin = (x_max - x_min) * 0.05
    ax.set_xlim(x_min - margin, x_max + margin)

    figure_path = os.path.join(current_path, "experiment_files", "export", "fig", "comparision_unsupervised", dataset,
                               transform_name)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    plt.tight_layout()  # <--- move this before savefig
    plt.savefig(os.path.join(figure_path, "comparision_detectors.pdf"))
    plt.savefig(os.path.join(figure_path, "comparision_detectors.pgf"))
    plt.tight_layout()
    plt.show()

    # %%
    # print default and best hyperparametrs for she dice and ash
    for detector in ["she", "dice", "ash_last", "nn_guided"]:
        detector_dir = os.path.join(base_results_dir, detector)
        params_path = os.path.join(detector_dir, "best_params.json")
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                best_params = json.load(f)
            print(f"\n=== Detector: {detector} ===")
            print("Best Parameters:")
            for k, v in best_params.items():
                print(f"  {k}: {v}")
        else:
            print(f"\n=== Detector: {detector} ===")
            print("No optimized parameters found.")
        # print default parameters
        default_params = get_default_ood_params(detector)
        print("Default Parameters:")
        for k, v in default_params.items():
            print(f"  {k}: {v}")
    # %%

    # mahalanobis is also far behind what i expcet maybe reimplemnt
    # %%
    import time
    import os
    import json
    import torch
    import numpy as np

    # ---------------------------------------------
    # SPEED TEST (compatible with TransformationProblem)
    # ---------------------------------------------
    speedtest_batches = 6
    speedtest_results_dir = os.path.join(base_results_dir, "speedtests")
    os.makedirs(speedtest_results_dir, exist_ok=True)

    print("\n=== Running speed tests for detectors (fixed) ===")
    for detector in detectors:
        if detector == "laplace_entropy_gridsearch":
            print(f"\n[SpeedTest] Skipping detector {detector} (too expensive).")
            continue
        print(f"\n[SpeedTest] Detector: {detector}")
        detector_dir = os.path.join(base_results_dir, detector)
        speedtest_path = os.path.join(speedtest_results_dir, f"{detector}_speed.json")
        if os.path.exists(speedtest_path):
            print(f"[SpeedTest] Found cached result at {speedtest_path}, skipping measurement.")
            continue

        # Pick params (optimized if available)
        params_path = os.path.join(detector_dir, "best_params.json")
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                best_params = json.load(f)
            print(f"[SpeedTest] Using optimized params.")
        else:
            best_params = get_default_ood_params(detector)
            print(f"[SpeedTest] Using default params.")

        # Try loading any saved model states
        best_model_path_prefix = os.path.join(detector_dir, "best_model")
        loaded_model_params = []
        idx = 0
        while os.path.exists(f"{best_model_path_prefix}_{idx}.pt"):
            loaded_model_params.append(torch.load(f"{best_model_path_prefix}_{idx}.pt", map_location=device))
            idx += 1

        # Build OOD problem
        final_kwargs = {
            "model": model,
            "train_cache": train_cache,
            "transform_seq": transform_seq_arg,
            "dataset_info": dataset_info,
            "architecture": architecture,
            "device": str(device),
            "val_id_loader": val_loader_preshuffled_in_dist,
            "val_ood_loader": val_loader_transformed_preshuffled,
        }
        if loaded_model_params:
            final_kwargs["model_params"] = loaded_model_params
        problem = create_ood_problem(detector, best_params, **final_kwargs)

        model.eval().to(device)
        torch.cuda.empty_cache()
        test_iter = iter(test_loader_transformed)

        times = []
        total_samples = 0
        with torch.no_grad():
            for batch_idx in range(speedtest_batches):
                try:
                    data, _ = next(test_iter)
                except StopIteration:
                    break
                data = data.to(device)
                total_samples += data.size(0)

                # Timing start
                torch.cuda.synchronize(device) if torch.cuda.is_available() else None
                start_time = time.perf_counter()

                # Simulate detector inference (similar to evaluate_confidence_and_search)
                res = optimizer_search_eval.optimize(problem, data, y=None)
                best_param = res[0] if isinstance(res, tuple) and len(res) >= 1 else res
                x_trans = problem.transform(data, best_param)
                _ = model(x_trans)

                # Timing end
                torch.cuda.synchronize(device) if torch.cuda.is_available() else None
                elapsed = time.perf_counter() - start_time

                # Skip first batch from timing
                if batch_idx > 0:
                    times.append(elapsed)
                    print(f"[SpeedTest] Batch {batch_idx + 1}: {elapsed:.4f}s")
                else:
                    print(f"[SpeedTest] Batch {batch_idx + 1} (warmup, ignored): {elapsed:.4f}s")

        # Summarize
        avg_batch_time = np.mean(times)
        avg_sample_time = avg_batch_time / (total_samples / len(times))
        result = {
            "detector": detector,
            "batches_tested": len(times),
            "avg_time_per_batch_sec": avg_batch_time,
            "avg_time_per_sample_sec": avg_sample_time,
            "device": str(device),
            "batch_size": dataset_info.batch_size,
        }
        with open(speedtest_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[SpeedTest] Saved results to {speedtest_path}")
        print(f"[SpeedTest] Avg: {avg_batch_time:.4f}s/batch ({avg_sample_time * 1e3:.3f} ms/sample)")

    print("\nAll speed tests completed successfully.")

    # %%
    import matplotlib.pyplot as plt

    # ---------------------------------------------
    # PLOT SPEED TEST RESULTS
    # ---------------------------------------------
    print("\n=== Plotting speed test results ===")

    speed_results = []
    for detector in detectors:
        speedtest_path = os.path.join(speedtest_results_dir, f"{detector}_speed.json")
        if os.path.exists(speedtest_path):
            with open(speedtest_path, "r") as f:
                data = json.load(f)
                speed_results.append(data)
        else:
            print(f"[Plot] Warning: No speed test result found for {detector}")

    if not speed_results:
        print("No speed test results found. Skipping plot.")
    else:
        detectors_plot = [r["detector"] for r in speed_results]
        per_sample_ms = [r["avg_time_per_sample_sec"] * 1000 for r in speed_results]
        per_batch_s = [r["avg_time_per_batch_sec"] for r in speed_results]

        # Sort by speed (ascending)
        sort_idx = np.argsort(per_sample_ms)
        detectors_plot = [detectors_plot[i] for i in sort_idx]
        per_sample_ms = [per_sample_ms[i] for i in sort_idx]
        per_batch_s = [per_batch_s[i] for i in sort_idx]

        plt.figure(figsize=(8, 5))
        plt.barh(detectors_plot, per_sample_ms)
        plt.xlabel("Avg Inference Time per Sample (ms)")
        plt.ylabel("Detector")
        plt.title("OOD Detector Speed Comparison")
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()

        plot_path = os.path.join(speedtest_results_dir, "detector_speed_comparison.png")
        plt.savefig(plot_path, dpi=200)
        plt.show()

        print(f"✅ Saved speed comparison plot to {plot_path}")

        # Optional: print a text summary
        print("\n=== Speed Summary (sorted by ms/sample) ===")
        for d, t_s, t_b in zip(detectors_plot, per_sample_ms, per_batch_s):
            print(f"{d:20s}  {t_s:8.3f} ms/sample   ({t_b:.4f} s/batch)")


def run_3_3():
    # %%
    import copy

    import torch
    import torch.nn as nn
    import torchvision
    import numpy as np
    from matplotlib import pyplot as plt

    from its.search import InverseTransformationSearch
    from search.parallel_gradient import ParallelGradientDescent
    from utils.affine_transforms_old import AffineTransformation2D
    from utils.sampling import BatchNegativeSampler

    # torch.cuda.is_available = lambda: False
    # device = torch.device("cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # look for experiment files in parents
    import os

    path_found = False
    current_path = os.getcwd()
    while not path_found:
        if os.path.exists(os.path.join(current_path, "experiment_files")):
            path_found = True
            break
        current_path = os.path.dirname(current_path)

    experiment_files_path_data = os.path.join(current_path, "experiment_files", "data")
    dataset = "modelnet10"

    default_architecutre_mapping = {
        "mnist": "resnet_small",
        "bigger_mnist": "resnet_small",
        "emnist": "extended_resnet_small",
        "bigger_emnist": "bigger_extended_resnet_small",
        "coil100": "coil_resnet_small",
        "tu_berlin": "bi_lstm",
        "modelnet10": "pointnetplus",
    }

    repeats = 5

    architecture = default_architecutre_mapping[dataset]
    budget = None
    # %%

    # %%
    from utils.transforms.apply import grid_resample_border, grid_resample_reflection
    # %%

    # %%
    from experiment_thesis.dataset_preperation.get_dataset import get_dataset_info, get_dataset

    dataset_info = get_dataset_info(dataset)

    dataset_dict = get_dataset(dataset_info, path=experiment_files_path_data, batch_size=dataset_info.batch_size)
    transform_name = dataset_info.transform_seq_name
    # %%

    dataset_dict.keys()
    dataset_train = dataset_dict['train_dataset']
    dataset_val = dataset_dict['val_dataset']
    dataset_test = dataset_dict['test_dataset']
    train_loader = dataset_dict['train_loader']
    val_loader = dataset_dict['val_loader']
    test_loader = dataset_dict['test_loader']
    n_classes = dataset_info.num_classes
    train_loader_transformed = dataset_dict['train_loader_transformed']
    val_loader_transformed = dataset_dict['val_loader_transformed']
    test_loader_transformed = dataset_dict['test_loader_transformed']
    train_loader_no_shuffle = dataset_dict['train_loader_no_shuffle']
    # %%

    # %%
    x = next(iter(test_loader_transformed))[0]
    # %%
    batch_size = next(iter(train_loader))[0].shape[0]

    # %%

    # %%
    from utils.eval.vis import vis_dataset, plt_setup_latex

    vis_dataset(train_loader, val_loader, test_loader_transformed)
    # %%
    from experiment_thesis.main import train_and_get_model, train_or_load_energy_model
    from experiment_thesis.dataset_preperation.basic_networks import get_network
    from utils.eval.main_model import evaluate_base_model

    model_dir_path = os.path.join(current_path, "experiment_files", "models")
    embedding_cache_path = os.path.join(current_path, "experiment_files", "embedding_cache")
    # Add results dir and helper for save paths
    results_dir_path = os.path.join(current_path, "experiment_files", "results", dataset, architecture,
                                    "comparison_supervised_methods")
    os.makedirs(results_dir_path, exist_ok=True)

    def savepath(label: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
        return os.path.join(results_dir_path, transform_name, f"{safe}.json")

    # %%
    model = get_network(dataset_info, architecture, num_classes=n_classes).to(device)
    modelname = f"{dataset}_{architecture}"
    cache_name_train = f"{dataset}_{architecture}_embedding_cache_train"

    train_and_get_model(model, model_dir_path, modelname, train_loader, val_loader, trainer_kwargs={
        "accelerator": "auto",
        "max_epochs": dataset_info.epochs,
        "precision": "16-mixed",
    }, load_if_exists=True)

    # %%
    model.eval().to(device)
    # %%
    # chek if data is image data
    is_image_data = len(dataset_info.input_size) == 3 and dataset_info.input_size[0] in [1, 3]
    # %%
    is_image_data = len(dataset_info.input_size) == 3 and dataset_info.input_size[0] in [1, 3]

    from utils.transforms.apply import grid_resample
    from experiment_thesis.dataset_preperation.transformation import get_transformation_sequence_images

    transform_seq = get_transformation_sequence_images(
        name=dataset_info.transform_seq_name,
        resample_method=dataset_info.resample_method,
        init_method="sobol"
    ).to(device)
    from utils.replacer import replace_rotation_transforms_2vec

    if dataset == "modelnet10":
        transform_seq = replace_rotation_transforms_2vec(transform_seq)
    # %%

    transform_seq_non_sobol = get_transformation_sequence_images(
        name=dataset_info.transform_seq_name,
        resample_method=dataset_info.resample_method,
    ).to(device)

    transform_seq_sobol = get_transformation_sequence_images(
        name=dataset_info.transform_seq_name,
        resample_method=dataset_info.resample_method,
        init_method="sobol"
    ).to(device)

    # %%
    from experiment_thesis.dataset_preperation.basic_networks import get_network_layer

    layer, layer_io = get_network_layer(dataset_info, architecture, 0, num_classes=None, num_rotations=8)
    # %%
    from confidence.direct.logit_based import EnergyConfidence
    from utils.transformation_problem import TransformationProblem
    from confidence.model.single_pass import SinglePassConfidence

    logit_energy = SinglePassConfidence(model, EnergyConfidence(), index=None)
    problem_energy_logits = TransformationProblem(logit_energy, transform_seq, consolidate_method="consolidate_simple")
    # test ot
    from search.shgo import SHGO
    optimizer = SHGO(initial_samples=46, local_runs=2, local_max_steps=3, local_opt_kwargs={"lr": 0.1})
    if dataset == "tu_berlin":
        optimizer = SHGO(initial_samples=60, local_runs=1, local_max_steps=0, local_opt_kwargs={"lr": 0.1})

    from utils.eval.ood_performance import load_or_run_evaluate_confidence_and_search, evaluate_confidence_and_search, \
        ITSWRAPPER

    # %%
    res = load_or_run_evaluate_confidence_and_search(
        model, optimizer=optimizer, problem=problem_energy_logits,
        test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
        save_path=savepath("energy_confidence_transformed"), show_progress=True,
        repeats=repeats)
    # %%
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # %%

    # %%

    # %%
    if dataset == "modelnet10":
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        def test_rotation_sampling(transform, n_samples=100000, domain=None):
            """Test and visualize rotation sampling distribution."""

            # Method 1: sample_param (individual)
            params_individual = transform.sample_param(batch_size=n_samples, domain=domain, device="cpu")

            # Method 2: sobol_to_param
            sobol_engine = torch.quasirandom.SobolEngine(dimension=3)
            sobol_samples = sobol_engine.draw(n_samples)
            params_sobol = transform.sobol_to_param(sobol_samples, domain)

            # Convert to rotation matrices and extract angles
            matrices_individual = transform.matrix(params_individual)
            matrices_sobol = transform.matrix(params_sobol)

            # Calculate rotation angles from trace
            def get_rotation_angles(matrices):
                traces = torch.einsum('...ii->...', matrices[..., :3, :3])
                cos_theta = torch.clamp((traces - 1.0) / 2.0, -1.0, 1.0)
                return torch.acos(cos_theta)

            angles_individual = get_rotation_angles(matrices_individual).numpy()
            angles_sobol = get_rotation_angles(matrices_sobol).numpy()

            # Theoretical PDF for uniform SO(3): (1/π)(1 - cos(θ))
            theta_range = np.linspace(0, np.pi, 200)
            pdf_theoretical = (1.0 / np.pi) * (1.0 - np.cos(theta_range))

            # Plot comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Individual sampling
            ax1.hist(angles_individual, bins=100, density=True, alpha=0.7, label='sample_param')
            ax1.plot(theta_range, pdf_theoretical, 'r-', linewidth=2, label='Theoretical SO(3)')
            ax1.set_xlabel(r'Rotation Angle (radians)')
            ax1.set_ylabel('Probability Density')
            ax1.set_title(f'{transform.__class__.__name__} - sample_param')
            ax1.legend()
            ax1.grid(True)

            # Sobol sampling
            ax2.hist(angles_sobol, bins=100, density=True, alpha=0.7, label='sobol_to_param')
            ax2.plot(theta_range, pdf_theoretical, 'r-', linewidth=2, label='Theoretical SO(3)')
            ax2.set_xlabel(r'Rotation Angle (radians)')
            ax2.set_ylabel('Probability Density')
            ax2.set_title(f'{transform.__class__.__name__} - sobol_to_param')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.show()

            # Statistical test
            print(f"\n{transform.__class__.__name__} Statistics:")
            print(f"sample_param - Mean angle: {angles_individual.mean():.4f}, Std: {angles_individual.std():.4f}")
            print(f"sobol_to_param - Mean angle: {angles_sobol.mean():.4f}, Std: {angles_sobol.std():.4f}")

        # Test both transforms
        from utils.transforms.rotation import Rotation3DEulerUniform, Rotation2Vec

        # To this (create an instance):
        test_rotation_sampling(Rotation3DEulerUniform(), n_samples=50000)

        test_rotation_sampling(Rotation2Vec(), n_samples=50000)
    # %%

    # %%

    # %%

    # %%
    from utils.augments import ComposeAugmentations, random_gaussian_noise, random_contrast, \
        random_gamma, random_blur_or_sharpen, build_default_augmentations
    import utils.augments

    @torch.no_grad()
    def dec_strat(x, idd, y_true):
        out = model(x)
        eq = out.argmax(dim=-1) == y_true
        # convert to tensor where y>=0 if correct, y<0 if incorrect
        y = torch.where(eq, y_true, -1)
        return y

    from utils.augments import build_default_augmentations, small_affine_augment_2d
    from utils.sampling_strategy import GaussianSamplingStrategyLatent, TransformLatentSamplingStrategy
    import importlib
    import utils.sampling_strategy
    import utils.sampling

    importlib.reload(utils.sampling)
    from utils.sampling import BatchNegativeSampler

    energy_model2 = get_network(dataset_info, architecture, num_classes=1).to(device)

    from experiment_thesis.main import train_or_load_energy_model

    if is_image_data:
        transform_true_function = small_affine_augment_2d
        multiple_augment = utils.augments.build_default_augmentations()
        affine_augment = ComposeAugmentations([
            lambda x: random_blur_or_sharpen(x, p=0.8, prob_blur=0.5,
                                             blur_ks_choices=(3, 5), blur_sigma_range=(0.2, 1.8),
                                             usm_ksize=5, usm_sigma_range=(0.5, 1.5),
                                             usm_amount_range=(0.5, 1.3), clamp=True),

        ])
    else:
        transform_true_function = None
        affine_augment = None
        multiple_augment = None

    negative_sampling_module = BatchNegativeSampler(
        TransformLatentSamplingStrategy(
            transform_sequence=transform_seq, ), transform_true_function
        =transform_true_function, augment_function=multiple_augment,
        decision_strategy=dec_strat,
    )

    energy_conf_default = train_or_load_energy_model(
        energy_model2, model_dir_path, f"{modelname}_energy2", train_loader,
        val_loader, trainer_kwargs={
            "accelerator": "auto",
            "max_epochs": dataset_info.epochs // 2,
            "precision": "16-mixed" if dataset_info.name not in ["modelnet10"] else "32",
        }, negative_sampling_module=negative_sampling_module, load_if_exists=True)

    energy_conf_default.to(device).eval()

    problem_energy_default = TransformationProblem(energy_conf_default, transform_seq,
                                                   consolidate_method="consolidate_simple")

    res = load_or_run_evaluate_confidence_and_search(
        model, optimizer=optimizer, problem=problem_energy_default,
        test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
        save_path=savepath("learned_energy_confidence_transformed"), show_progress=True,
        repeats=repeats)

    # unload values
    del energy_model2
    del energy_conf_default
    del problem_energy_default
    print(res)

    # %%
    from utils.augments import ComposeAugmentations, random_gaussian_noise, random_contrast, \
        random_gamma, random_blur_or_sharpen, build_default_augmentations
    import utils.augments

    @torch.no_grad()
    def dec_strat(x, idd, y_true):
        out = model(x)
        eq = out.argmax(dim=-1) == y_true
        # convert to tensor where y>=0 if correct, y<0 if incorrect
        y = torch.where(eq, y_true, -1)
        return y

    from utils.augments import build_default_augmentations, small_affine_augment_2d
    from utils.sampling_strategy import GaussianSamplingStrategyLatent, TransformLatentSamplingStrategy
    import importlib
    import utils.sampling_strategy
    import utils.sampling

    importlib.reload(utils.sampling)
    from utils.sampling import BatchNegativeSampler

    energy_model2 = get_network(dataset_info, architecture, num_classes=1).to(device)

    from experiment_thesis.main import train_or_load_energy_model

    if is_image_data:
        transform_true_function = small_affine_augment_2d
        multiple_augment = utils.augments.build_default_augmentations()
        affine_augment = ComposeAugmentations([
            lambda x: random_blur_or_sharpen(x, p=0.8, prob_blur=0.5,
                                             blur_ks_choices=(3, 5), blur_sigma_range=(0.2, 1.8),
                                             usm_ksize=5, usm_sigma_range=(0.5, 1.5),
                                             usm_amount_range=(0.5, 1.3), clamp=True),

        ])
    else:
        transform_true_function = None
        affine_augment = None
        multiple_augment = None

    negative_sampling_module = BatchNegativeSampler(
        TransformLatentSamplingStrategy(
            transform_sequence=transform_seq, ), transform_true_function
        =transform_true_function, augment_function=multiple_augment,
        decision_strategy=dec_strat,
    )

    energy_conf_default = train_or_load_energy_model(
        energy_model2, model_dir_path, f"{modelname}_energy2_det", train_loader,
        val_loader, trainer_kwargs={
            "accelerator": "auto",
            "max_epochs": dataset_info.epochs // 2,
            "precision": "16-mixed" if dataset_info.name not in ["modelnet10"] else "32",
        }, negative_sampling_module=negative_sampling_module, load_if_exists=True, deterministic_val=True)

    energy_conf_default.to(device).eval()

    problem_energy_default = TransformationProblem(energy_conf_default, transform_seq,
                                                   consolidate_method="consolidate_simple")

    res = load_or_run_evaluate_confidence_and_search(
        model, optimizer=optimizer, problem=problem_energy_default,
        test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
        save_path=savepath("learned_energy_confidence_transformed_det"), show_progress=True,
        repeats=repeats)

    # unload values
    del energy_model2
    del energy_conf_default
    del problem_energy_default
    print(res)

    # %%
    from experiment_thesis.dataset_preperation.basic_networks import make_deterministic
    from utils.augments import ComposeAugmentations, random_gaussian_noise, random_contrast, \
        random_gamma, random_blur_or_sharpen, build_default_augmentations
    import utils.augments

    if dataset == "modelnet10":
        @torch.no_grad()
        def dec_strat_model(x, idd, y_true):
            make_deterministic(model, random=True, verbose=False)
            out = model(x)
            make_deterministic(model, random=False, verbose=False)
            eq = out.argmax(dim=-1) == y_true
            # convert to tensor where y>=0 if correct, y<0 if incorrect
            y = torch.where(eq, y_true, -1)
            return y

        from utils.augments import build_default_augmentations, small_affine_augment_2d
        from utils.sampling_strategy import GaussianSamplingStrategyLatent, TransformLatentSamplingStrategy
        import importlib
        import utils.sampling_strategy
        import utils.sampling

        importlib.reload(utils.sampling)
        from utils.sampling import BatchNegativeSampler

        energy_model4 = get_network(dataset_info, architecture, num_classes=1).to(device)

        from experiment_thesis.main import train_or_load_energy_model

        if is_image_data:
            transform_true_function = small_affine_augment_2d
            multiple_augment = utils.augments.build_default_augmentations()
            affine_augment = ComposeAugmentations([
                lambda x: random_blur_or_sharpen(x, p=0.8, prob_blur=0.5,
                                                 blur_ks_choices=(3, 5), blur_sigma_range=(0.2, 1.8),
                                                 usm_ksize=5, usm_sigma_range=(0.5, 1.5),
                                                 usm_amount_range=(0.5, 1.3), clamp=True),

            ])
        else:
            transform_true_function = None
            affine_augment = None
            multiple_augment = None

        negative_sampling_module = BatchNegativeSampler(
            TransformLatentSamplingStrategy(
                transform_sequence=transform_seq, ), transform_true_function
            =transform_true_function, augment_function=multiple_augment,
            decision_strategy=dec_strat_model,
        )

        energy_conf_default = train_or_load_energy_model(
            energy_model4, model_dir_path, f"{modelname}_energy4", train_loader,
            val_loader, trainer_kwargs={
                "accelerator": "auto",
                "max_epochs": dataset_info.epochs // 2,
                "precision": "16-mixed" if dataset_info.name not in ["modelnet10"] else "32",
            }, negative_sampling_module=negative_sampling_module, load_if_exists=True)

        energy_conf_default.to(device).eval()

        problem_energy_default = TransformationProblem(energy_conf_default, transform_seq,
                                                       consolidate_method="consolidate_simple")

        res = load_or_run_evaluate_confidence_and_search(
            model, optimizer=optimizer, problem=problem_energy_default,
            test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
            save_path=savepath("learned_energy_confidence_transformed4"), show_progress=True,
            repeats=repeats)

        # unload values
        del energy_model4
        del energy_conf_default
        del problem_energy_default
        print(res)

    # %%
    W = plt_setup_latex()
    # %%
    figure_path = os.path.join(current_path, "experiment_files", "export", "fig", "comparision_supervised", dataset,
                               transform_name)
    os.makedirs(figure_path, exist_ok=True)
    # %%
    if is_image_data:
        x = next(iter(test_loader))[0]
        # take only 72
        x = x[:72]

        x_aug = transform_true_function(x)
        dif = (x - x_aug).abs()

        fig, axs = plt.subplots(1, 3, figsize=(W, W / 2))  # 1 row, 3 columns

        # Original
        axs[0].set_title("Original")
        axs[0].imshow(torchvision.utils.make_grid(x, nrow=8).cpu().permute(1, 2, 0))
        axs[0].axis("off")

        # Augmented
        axs[1].set_title("Augmented")
        axs[1].imshow(torchvision.utils.make_grid(x_aug, nrow=8).cpu().permute(1, 2, 0))
        axs[1].axis("off")

        # Difference
        axs[2].set_title("Difference")
        axs[2].imshow(torchvision.utils.make_grid(dif, nrow=8).cpu().permute(1, 2, 0))
        axs[2].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(figure_path, "true_augmentations.png"), dpi=300)

        plt.show()

    # %%
    if is_image_data:
        x = next(iter(test_loader))[0]
        # take only 72
        x = x[:72]

        x_aug = affine_augment(x)
        dif = (x - x_aug).abs()

        fig, axs = plt.subplots(1, 3, figsize=(W, W / 2))  # 1 row, 3 columns

        # Original
        axs[0].set_title("Original")
        axs[0].imshow(torchvision.utils.make_grid(x, nrow=8).cpu().permute(1, 2, 0))
        axs[0].axis("off")

        # Augmented
        axs[1].set_title("Augmented")
        axs[1].imshow(torchvision.utils.make_grid(x_aug, nrow=8).cpu().permute(1, 2, 0))
        axs[1].axis("off")

        # Difference
        axs[2].set_title("Difference")
        axs[2].imshow(torchvision.utils.make_grid(dif, nrow=8).cpu().permute(1, 2, 0))
        axs[2].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(figure_path, "affine_augmentations.png"), dpi=300)
        plt.show()

    # %%

    # %%
    # if image data also test a variant without augmentations
    if is_image_data:
        energy_model_no_aug = get_network(dataset_info, architecture, num_classes=1).to(device)
        negative_sampling_module_no_aug = BatchNegativeSampler(
            TransformLatentSamplingStrategy(
                transform_sequence=transform_seq, ), transform_true_function
            =None, augment_function=None,
            decision_strategy=dec_strat,
        )

        energy_conf_no_aug = train_or_load_energy_model(
            copy.deepcopy(energy_model_no_aug), model_dir_path, f"{modelname}_energy2_no_aug", train_loader,
            val_loader, trainer_kwargs={
                "accelerator": "auto",
                "max_epochs": dataset_info.epochs // 2,
                "precision": "16-mixed" if dataset_info.name not in ["modelnet10"] else "32",
            }, negative_sampling_module=negative_sampling_module_no_aug, load_if_exists=True)
        energy_conf_no_aug.to(device).eval()
        problem_energy_no_aug = TransformationProblem(energy_conf_no_aug, transform_seq,
                                                      consolidate_method="consolidate_simple")

        res = load_or_run_evaluate_confidence_and_search(
            model, optimizer=optimizer, problem=problem_energy_no_aug,
            test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
            save_path=savepath("learned_energy_confidence_no_aug_transformed"), show_progress=True,
            repeats=repeats)

        # unload values
        del energy_model_no_aug
        del energy_conf_no_aug
        del problem_energy_no_aug
        print(res)
    # %%
    # if image data also test a variant with only blur augmentations
    if is_image_data:
        energy_model_no_aug = get_network(dataset_info, architecture, num_classes=1).to(device)
        negative_sampling_module_no_aug = BatchNegativeSampler(
            TransformLatentSamplingStrategy(
                transform_sequence=transform_seq, mode="double_resampled"), transform_true_function
            =transform_true_function, augment_function=affine_augment,
            decision_strategy=dec_strat,
        )

        energy_conf_no_aug = train_or_load_energy_model(
            copy.deepcopy(energy_model_no_aug), model_dir_path, f"{modelname}_energy2_only_blur_aug_double_rs",
            train_loader,
            val_loader, trainer_kwargs={
                "accelerator": "auto",
                "max_epochs": dataset_info.epochs // 2,
                "precision": "16-mixed" if dataset_info.name not in ["modelnet10"] else "32",
            }, negative_sampling_module=negative_sampling_module_no_aug, load_if_exists=True)
        energy_conf_no_aug.to(device).eval()
        problem_energy_no_aug = TransformationProblem(energy_conf_no_aug, transform_seq,
                                                      consolidate_method="consolidate_simple")

        res = load_or_run_evaluate_confidence_and_search(
            model, optimizer=optimizer, problem=problem_energy_no_aug,
            test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
            save_path=savepath("learned_energy_confidence_only_blur_aug_transformed_double_rs"), show_progress=True,
            repeats=repeats)

        # unload values
        del energy_model_no_aug
        del energy_conf_no_aug
        del problem_energy_no_aug
        print(res)
    # %%
    # if image data also test a variant with only blur augmentations
    if is_image_data:
        energy_model_no_aug = get_network(dataset_info, architecture, num_classes=1).to(device)
        negative_sampling_module_no_aug = BatchNegativeSampler(
            TransformLatentSamplingStrategy(
                transform_sequence=transform_seq, mode="double"), transform_true_function
            =transform_true_function, augment_function=affine_augment,
            decision_strategy=dec_strat,
        )

        energy_conf_no_aug = train_or_load_energy_model(
            copy.deepcopy(energy_model_no_aug), model_dir_path, f"{modelname}_energy2_only_blur_aug_double",
            train_loader,
            val_loader, trainer_kwargs={
                "accelerator": "auto",
                "max_epochs": dataset_info.epochs // 2,
                "precision": "16-mixed" if dataset_info.name not in ["modelnet10"] else "32",
            }, negative_sampling_module=negative_sampling_module_no_aug, load_if_exists=True)
        energy_conf_no_aug.to(device).eval()
        problem_energy_no_aug = TransformationProblem(energy_conf_no_aug, transform_seq,
                                                      consolidate_method="consolidate_simple")

        res = load_or_run_evaluate_confidence_and_search(
            model, optimizer=optimizer, problem=problem_energy_no_aug,
            test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
            save_path=savepath("learned_energy_confidence_only_blur_aug_transformed_double"), show_progress=True,
            repeats=repeats)

        # unload values
        del energy_model_no_aug
        del energy_conf_no_aug
        del problem_energy_no_aug
        print(res)
    # %%

    # %%
    # if image data also test a variant with only blur augmentations
    if is_image_data:
        energy_model_no_aug = get_network(dataset_info, architecture, num_classes=1).to(device)
        negative_sampling_module_no_aug = BatchNegativeSampler(
            TransformLatentSamplingStrategy(
                transform_sequence=transform_seq, ), transform_true_function
            =transform_true_function, augment_function=affine_augment,
            decision_strategy=dec_strat,
        )

        energy_conf_no_aug = train_or_load_energy_model(
            copy.deepcopy(energy_model_no_aug), model_dir_path, f"{modelname}_energy2_only_blur_aug", train_loader,
            val_loader, trainer_kwargs={
                "accelerator": "auto",
                "max_epochs": dataset_info.epochs // 2,
                "precision": "16-mixed" if dataset_info.name not in ["modelnet10"] else "32",
            }, negative_sampling_module=negative_sampling_module_no_aug, load_if_exists=True)
        energy_conf_no_aug.to(device).eval()
        problem_energy_no_aug = TransformationProblem(energy_conf_no_aug, transform_seq,
                                                      consolidate_method="consolidate_simple")

        res = load_or_run_evaluate_confidence_and_search(
            model, optimizer=optimizer, problem=problem_energy_no_aug,
            test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
            save_path=savepath("learned_energy_confidence_only_blur_aug_transformed"), show_progress=True,
            repeats=repeats)

        # unload values
        del energy_model_no_aug
        del energy_conf_no_aug
        del problem_energy_no_aug
        print(res)
    # %%
    # if image data also test a variant without augmentations
    if is_image_data:
        energy_model_true_aug_only = get_network(dataset_info, architecture, num_classes=1).to(device)
        negative_sampling_module_true_aug_only = BatchNegativeSampler(
            TransformLatentSamplingStrategy(
                transform_sequence=transform_seq, ), transform_true_function
            =transform_true_function, augment_function=None,
            decision_strategy=dec_strat,
        )
        energy_conf_true_aug_only = train_or_load_energy_model(
            copy.deepcopy(energy_model_true_aug_only), model_dir_path, f"{modelname}_energy2_true_aug_only",
            train_loader,
            val_loader, trainer_kwargs={
                "accelerator": "auto",
                "max_epochs": dataset_info.epochs // 2,
                "precision": "16-mixed" if dataset_info.name not in ["modelnet10"] else "32",
            }, negative_sampling_module=negative_sampling_module_true_aug_only, load_if_exists=True)
        energy_conf_true_aug_only.to(device).eval()
        problem_energy_true_aug_only = TransformationProblem(energy_conf_true_aug_only, transform_seq,
                                                             consolidate_method="consolidate_simple")
        res = load_or_run_evaluate_confidence_and_search(
            model, optimizer=optimizer, problem=problem_energy_true_aug_only,
            test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
            save_path=savepath("learned_energy_confidence_true_aug_only_transformed"), show_progress=True,
            repeats=repeats)
        # unload values
        del energy_model_true_aug_only
        del energy_conf_true_aug_only
        del problem_energy_true_aug_only
        print(res)
    # %%
    if is_image_data:
        energy_model_no_true_aug = get_network(dataset_info, architecture, num_classes=1).to(device)
        negative_sampling_module_no_true_aug = BatchNegativeSampler(
            TransformLatentSamplingStrategy(
                transform_sequence=transform_seq, ), transform_true_function
            =None, augment_function=affine_augment,
            decision_strategy=dec_strat,
        )
        energy_conf_no_true_aug = train_or_load_energy_model(
            copy.deepcopy(energy_model_no_true_aug), model_dir_path, f"{modelname}_energy2_no_true_aug", train_loader,
            val_loader, trainer_kwargs={
                "accelerator": "auto",
                "max_epochs": dataset_info.epochs // 2,
                "precision": "16-mixed" if dataset_info.name not in ["modelnet10"] else "32",
            }, negative_sampling_module=negative_sampling_module_no_true_aug, load_if_exists=True)
        energy_conf_no_true_aug.to(device).eval()
        problem_energy_no_true_aug = TransformationProblem(energy_conf_no_true_aug, transform_seq,
                                                           consolidate_method="consolidate_simple")
        res = load_or_run_evaluate_confidence_and_search(
            model, optimizer=optimizer, problem=problem_energy_no_true_aug,
            test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
            save_path=savepath("learned_energy_confidence_no_true_aug_transformed"), show_progress=True,
            repeats=repeats)
        # unload values
        del energy_model_no_true_aug
        del energy_conf_no_true_aug
        del problem_energy_no_true_aug
        print(res)
    # %%
    import pandas as pd
    import json
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    if is_image_data:
        res_all_augment = json.load(open(savepath("learned_energy_confidence_transformed"), "r"))
        res_no_augment = json.load(open(savepath("learned_energy_confidence_no_aug_transformed"), "r"))
        res_blur_augment = json.load(open(savepath("learned_energy_confidence_only_blur_aug_transformed"), "r"))
        res_true_augment_only = json.load(open(savepath("learned_energy_confidence_true_aug_only_transformed"), "r"))
        res_blur_no_true = json.load(open(savepath("learned_energy_confidence_no_true_aug_transformed"), "r"))

        # Prepare data
        augment_types = [
            "No Augment",
            "True",
            "Blur",
            "Both",
            "Additional"
        ]
        means = [
            res_no_augment['accuracy_mean'],
            res_true_augment_only['accuracy_mean'],
            res_blur_no_true['accuracy_mean'],
            res_blur_augment['accuracy_mean'],
            res_all_augment['accuracy_mean']
        ]
        ses = [
            res_no_augment.get('accuracy_se', 0),
            res_true_augment_only.get('accuracy_se', 0),
            res_blur_no_true.get('accuracy_se', 0),
            res_blur_augment.get('accuracy_se', 0),
            res_all_augment.get('accuracy_se', 0)
        ]

        # Plotting
        plt.figure(figsize=(W, W * 0.5))
        ax = sns.barplot(x=augment_types, y=means, palette="tab10")

        # Add error bars
        ax.errorbar(x=np.arange(len(augment_types)), y=means, yerr=ses, fmt='none', c='black', capsize=5)

        # Dynamic y-limits based on mean ± SE
        ymin = min([m - s for m, s in zip(means, ses)]) - 0.05
        ymax = max([m + s for m, s in zip(means, ses)]) + 0.05
        ax.set_ylim(ymin, ymax)

        ax.set_ylabel("Accuracy")
        # plt.title("Comparison of Different Augmentation Strategies")
        plt.savefig(os.path.join(figure_path, "augmentation_strategies_comparison.pgf"), dpi=300)
        plt.savefig(os.path.join(figure_path, "augmentation_strategies_comparison.pdf"))
        plt.show()

    # %%
    # now test the variant that uses no dec strat
    energy_model_no_dec_strat = get_network(dataset_info, architecture, num_classes=1).to(device)
    negative_sampling_module_no_dec_strat = BatchNegativeSampler(
        TransformLatentSamplingStrategy(
            transform_sequence=transform_seq, ), transform_true_function
        =transform_true_function, augment_function=affine_augment,
        decision_strategy=None,
    )
    energy_conf_no_dec_strat = train_or_load_energy_model(
        copy.deepcopy(energy_model_no_dec_strat), model_dir_path, f"{modelname}_energy2_no_dec_strat", train_loader,
        val_loader, trainer_kwargs={
            "accelerator": "auto",
            "max_epochs": dataset_info.epochs // 2,
            "precision": "16-mixed" if dataset_info.name not in ["modelnet10"] else "32",
        }, negative_sampling_module=negative_sampling_module_no_dec_strat, load_if_exists=True)
    energy_conf_no_dec_strat.to(device).eval()
    problem_energy_no_dec_strat = TransformationProblem(energy_conf_no_dec_strat, transform_seq,
                                                        consolidate_method="consolidate_simple")
    res = load_or_run_evaluate_confidence_and_search(
        model, optimizer=optimizer, problem=problem_energy_no_dec_strat,
        test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
        save_path=savepath("learned_energy_confidence_no_dec_strat_transformed"), show_progress=True,
        repeats=repeats)
    del energy_model_no_dec_strat
    del energy_conf_no_dec_strat
    del problem_energy_no_dec_strat
    print(res)
    # %%

    # %%
    # now no dec strat no augmentations
    if is_image_data:
        energy_model_no_dec_strat_no_aug = get_network(dataset_info, architecture, num_classes=1).to(device)
        negative_sampling_module_no_dec_strat_no_aug = BatchNegativeSampler(
            TransformLatentSamplingStrategy(
                transform_sequence=transform_seq, ), transform_true_function
            =None, augment_function=None,
            decision_strategy=None,
        )
        energy_conf_no_dec_strat_no_aug = train_or_load_energy_model(
            copy.deepcopy(energy_model_no_dec_strat_no_aug), model_dir_path, f"{modelname}_energy2_no_dec_strat_no_aug",
            train_loader,
            val_loader, trainer_kwargs={
                "accelerator": "auto",
                "max_epochs": dataset_info.epochs // 2,
                "precision": "16-mixed" if dataset_info.name not in ["modelnet10"] else "32",
            }, negative_sampling_module=negative_sampling_module_no_dec_strat_no_aug, load_if_exists=True)
        energy_conf_no_dec_strat_no_aug.to(device).eval()
        problem_energy_no_dec_strat_no_aug = TransformationProblem(energy_conf_no_dec_strat_no_aug, transform_seq,
                                                                   consolidate_method="consolidate_simple")
        res = load_or_run_evaluate_confidence_and_search(
            model, optimizer=optimizer, problem=problem_energy_no_dec_strat_no_aug,
            test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
            save_path=savepath("learned_energy_confidence_no_dec_strat_no_aug_transformed"), show_progress=True,
            repeats=repeats)
        del energy_model_no_dec_strat_no_aug
        del energy_conf_no_dec_strat_no_aug
        del problem_energy_no_dec_strat_no_aug
        print(res)

    # %%
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    if is_image_data:
        res_blur_augment = json.load(open(savepath("learned_energy_confidence_only_blur_aug_transformed"), "r"))
        res_no_dec_strat = json.load(open(savepath("learned_energy_confidence_no_dec_strat_transformed"), "r"))
        res_no_dec_strat_no_aug = json.load(
            open(savepath("learned_energy_confidence_no_dec_strat_no_aug_transformed"), "r"))

        loss_types = [
            "Model-based",
            "Data-based",
            "Data-no-aug"
        ]
        means = [
            res_blur_augment['accuracy_mean'],
            res_no_dec_strat['accuracy_mean'],
            res_no_dec_strat_no_aug['accuracy_mean']
        ]
        ses = [
            res_blur_augment.get('accuracy_se', 0),
            res_no_dec_strat.get('accuracy_se', 0),
            res_no_dec_strat_no_aug.get('accuracy_se', 0)
        ]

    elif dataset == "modelnet10":
        res_dec_strat = json.load(open(savepath("learned_energy_confidence_transformed"), "r"))
        res_dec_strat_train = json.load(open(savepath("learned_energy_confidence_transformed4"), "r"))
        res_no_dec_strat = json.load(open(savepath("learned_energy_confidence_no_dec_strat_transformed"), "r"))

        loss_types = [
            "Model-based",
            "Model-based (sampled)",
            "Data-based"
        ]
        means = [
            res_dec_strat['accuracy_mean'],
            res_dec_strat_train['accuracy_mean'],
            res_no_dec_strat['accuracy_mean']
        ]
        ses = [
            res_dec_strat.get('accuracy_se', 0),
            res_dec_strat_train.get('accuracy_se', 0),
            res_no_dec_strat.get('accuracy_se', 0)
        ]
    else:
        res_dec_strat = json.load(open(savepath("learned_energy_confidence_transformed"), "r"))
        res_no_dec_strat = json.load(open(savepath("learned_energy_confidence_no_dec_strat_transformed"), "r"))

        loss_types = [
            "Model-based",
            "Data-based"
        ]
        means = [
            res_dec_strat['accuracy_mean'],
            res_no_dec_strat['accuracy_mean'],

        ]
        ses = [
            res_dec_strat.get('accuracy_se', 0),
            res_no_dec_strat.get('accuracy_se', 0),
        ]

    # Plotting
    plt.figure(figsize=(W, W / 2))
    ax = sns.barplot(x=loss_types, y=means, palette="tab10")

    # Add error bars
    ax.errorbar(x=np.arange(len(loss_types)), y=means, yerr=ses, fmt='none', c='black', capsize=5)

    # Dynamic y-limits based on mean ± SE
    ymin = min([m - s for m, s in zip(means, ses)]) - 0.05
    ymax = max([m + s for m, s in zip(means, ses)]) + 0.05
    ax.set_ylim(ymin, ymax)

    ax.set_ylabel("Accuracy")
    # plt.title("Comparison of Different Strategies")
    plt.savefig(os.path.join(figure_path, "dec_strat_vs_no_dec_strat.pdf"), dpi=300)
    plt.savefig(os.path.join(figure_path, "dec_strat_vs_no_dec_strat.pgf"), dpi=300)
    plt.show()

    # %%

    # %%
    # if image data also test a variant with only blur augmentations
    if is_image_data:
        energy_model_no_aug_mse = get_network(dataset_info, architecture, num_classes=1).to(device)
        negative_sampling_module_no_aug = BatchNegativeSampler(
            TransformLatentSamplingStrategy(
                transform_sequence=transform_seq, ), transform_true_function
            =transform_true_function, augment_function=affine_augment,
            decision_strategy=dec_strat,
        )

        energy_conf_no_aug = train_or_load_energy_model(
            copy.deepcopy(energy_model_no_aug_mse), model_dir_path, f"{modelname}_energy2_only_blur_aug_mse_loss",
            train_loader,
            val_loader, trainer_kwargs={
                "accelerator": "auto",
                "max_epochs": dataset_info.epochs // 2,
                "precision": "16-mixed" if dataset_info.name not in ["modelnet10"] else "32",
            }, negative_sampling_module=negative_sampling_module_no_aug, load_if_exists=True, loss_type="mse")
        energy_conf_no_aug.to(device).eval()
        problem_energy_no_aug = TransformationProblem(energy_conf_no_aug, transform_seq,
                                                      consolidate_method="consolidate_simple")

        res = load_or_run_evaluate_confidence_and_search(
            model, optimizer=optimizer, problem=problem_energy_no_aug,
            test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
            save_path=savepath("learned_energy_confidence_only_blur_aug_transformed_mse_loss"), show_progress=True,
            repeats=repeats)

        # unload values
        del energy_model_no_aug_mse
        del energy_conf_no_aug
        del problem_energy_no_aug
        print(res)
    # %%

    # %%
    # now with discrimator loss
    if is_image_data:
        energy_model_no_aug_disc = get_network(dataset_info, architecture, num_classes=1).to(device)
        negative_sampling_module_no_aug = BatchNegativeSampler(
            TransformLatentSamplingStrategy(
                transform_sequence=transform_seq, ), transform_true_function
            =transform_true_function, augment_function=affine_augment,
            decision_strategy=dec_strat,
        )

        energy_conf_no_aug_disc = train_or_load_energy_model(
            copy.deepcopy(energy_model_no_aug_disc), model_dir_path, f"{modelname}_energy2_only_blur_aug_disc_loss",
            train_loader,
            val_loader, trainer_kwargs={
                "accelerator": "auto",
                "max_epochs": dataset_info.epochs // 2,
                "precision": "32",
                "detect_anomaly": False,
            }, negative_sampling_module=negative_sampling_module_no_aug, load_if_exists=True, loss_type="discriminator",
            optimizer_type=torch.optim.AdamW, gp_weight=5)
        energy_conf_no_aug_disc.to(device).eval()
        problem_energy_no_aug_disc = TransformationProblem(energy_conf_no_aug_disc, transform_seq,
                                                           consolidate_method="consolidate_simple")

        res = load_or_run_evaluate_confidence_and_search(
            model, optimizer=optimizer, problem=problem_energy_no_aug_disc,
            test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
            save_path=savepath("learned_energy_confidence_only_blur_aug_transformed_disc_loss"), show_progress=True,
            repeats=repeats)

        # unload values
        del energy_model_no_aug_disc
        del energy_conf_no_aug_disc
        del problem_energy_no_aug_disc
        print(res)
    # %%
    # now with discrimantitro lsos and no dec strat
    if is_image_data:
        energy_model_no_aug_disc_no_dec_strat = get_network(dataset_info, architecture, num_classes=1).to(device)
        negative_sampling_module_no_aug_no_dec_strat = BatchNegativeSampler(
            TransformLatentSamplingStrategy(
                transform_sequence=transform_seq, ), transform_true_function
            =transform_true_function, augment_function=affine_augment,
            decision_strategy=None,
        )

        energy_conf_no_aug_disc_no_dec_strat = train_or_load_energy_model(
            copy.deepcopy(energy_model_no_aug_disc_no_dec_strat), model_dir_path,
            f"{modelname}_energy2_only_blur_aug_disc_loss_no_dec_strat", train_loader,
            val_loader, trainer_kwargs={
                "accelerator": "auto",
                "max_epochs": dataset_info.epochs // 2,
                "precision": "32",
            }, negative_sampling_module=negative_sampling_module_no_aug_no_dec_strat, load_if_exists=True,
            loss_type="discriminator", optimizer_type=torch.optim.AdamW, gp_weight=5)
        energy_conf_no_aug_disc_no_dec_strat.to(device).eval()
        problem_energy_no_aug_disc_no_dec_strat = TransformationProblem(energy_conf_no_aug_disc_no_dec_strat,
                                                                        transform_seq,
                                                                        consolidate_method="consolidate_simple")

        res = load_or_run_evaluate_confidence_and_search(
            model, optimizer=optimizer, problem=problem_energy_no_aug_disc_no_dec_strat,
            test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
            save_path=savepath("learned_energy_confidence_only_blur_aug_transformed_disc_loss_no_dec_strat"),
            show_progress=True,
            repeats=repeats)
        # unload values
        del energy_model_no_aug_disc_no_dec_strat
        del energy_conf_no_aug_disc_no_dec_strat
        del problem_energy_no_aug_disc_no_dec_strat
        print(res)
    # %%
    # now bce loss with adamw
    if is_image_data:
        energy_model_no_aug_bce = get_network(dataset_info, architecture, num_classes=1).to(device)
        negative_sampling_module_no_aug_bce = BatchNegativeSampler(
            TransformLatentSamplingStrategy(
                transform_sequence=transform_seq, ), transform_true_function
            =transform_true_function, augment_function=affine_augment,
            decision_strategy=dec_strat,
        )

        energy_conf_no_aug_bce = train_or_load_energy_model(
            copy.deepcopy(energy_model_no_aug_bce), model_dir_path, f"{modelname}_energy2_only_blur_aug_bce_loss",
            train_loader,
            val_loader, trainer_kwargs={
                "accelerator": "auto",
                "max_epochs": dataset_info.epochs // 2,
                "precision": "16-mixed" if dataset_info.name not in ["modelnet10"] else "32",
            }, negative_sampling_module=negative_sampling_module_no_aug_bce, load_if_exists=True, loss_type="bce",
            optimizer_type=torch.optim.AdamW)
        energy_conf_no_aug_bce.to(device).eval()
        problem_energy_no_aug_bce = TransformationProblem(energy_conf_no_aug_bce, transform_seq,
                                                          consolidate_method="consolidate_simple")

        res = load_or_run_evaluate_confidence_and_search(
            model, optimizer=optimizer, problem=problem_energy_no_aug_bce,
            test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
            save_path=savepath("learned_energy_confidence_only_blur_aug_transformed_bce_loss"), show_progress=True,
            repeats=repeats)

        # unload values
        del energy_model_no_aug_bce
        del energy_conf_no_aug_bce
        del problem_energy_no_aug_bce
        print(res)
    # %%
    # now margin and triplet loss(dont support dec strat)
    if is_image_data:
        energy_model_no_aug_margin = get_network(dataset_info, architecture, num_classes=1).to(device)
        negative_sampling_module_no_aug_margin = BatchNegativeSampler(
            TransformLatentSamplingStrategy(
                transform_sequence=transform_seq, ), transform_true_function
            =transform_true_function, augment_function=affine_augment,
            decision_strategy=None,
        )

        energy_conf_no_aug_margin = train_or_load_energy_model(
            copy.deepcopy(energy_model_no_aug_margin), model_dir_path, f"{modelname}_energy2_only_blur_aug_margin_loss",
            train_loader,
            val_loader, trainer_kwargs={
                "accelerator": "auto",
                "max_epochs": dataset_info.epochs // 2,
                "precision": "16-mixed" if dataset_info.name not in ["modelnet10"] else "32",
            }, negative_sampling_module=negative_sampling_module_no_aug_margin, load_if_exists=True, loss_type="margin",
            monitor="val_auroc")
        energy_conf_no_aug_margin.to(device).eval()
        problem_energy_no_aug_margin = TransformationProblem(energy_conf_no_aug_margin, transform_seq,
                                                             consolidate_method="consolidate_simple")

        res = load_or_run_evaluate_confidence_and_search(
            model, optimizer=optimizer, problem=problem_energy_no_aug_margin,
            test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
            save_path=savepath("learned_energy_confidence_only_blur_aug_transformed_margin_loss"), show_progress=True,
            repeats=repeats)

        # unload values
        del energy_model_no_aug_margin
        del energy_conf_no_aug_margin
        del problem_energy_no_aug_margin
        print(res)
    # %%
    if is_image_data:
        energy_model_no_aug_triplet = get_network(dataset_info, architecture, num_classes=1).to(device)
        negative_sampling_module_no_aug_triplet = BatchNegativeSampler(
            TransformLatentSamplingStrategy(
                transform_sequence=transform_seq, ), transform_true_function
            =transform_true_function, augment_function=affine_augment,
            decision_strategy=None,
        )

        energy_conf_no_aug_triplet = train_or_load_energy_model(
            copy.deepcopy(energy_model_no_aug_triplet), model_dir_path,
            f"{modelname}_energy2_only_blur_aug_triplet_loss", train_loader,
            val_loader, trainer_kwargs={
                "accelerator": "auto",
                "max_epochs": dataset_info.epochs // 2,
                "precision": "16-mixed" if dataset_info.name not in ["modelnet10"] else "32",
            }, negative_sampling_module=negative_sampling_module_no_aug_triplet, load_if_exists=True,
            loss_type="triplet", monitor="val_smaller")
        energy_conf_no_aug_triplet.to(device).eval()
        problem_energy_no_aug_triplet = TransformationProblem(energy_conf_no_aug_triplet, transform_seq,
                                                              consolidate_method="consolidate_simple")

        res = load_or_run_evaluate_confidence_and_search(
            model, optimizer=optimizer, problem=problem_energy_no_aug_triplet,
            test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
            save_path=savepath("learned_energy_confidence_only_blur_aug_transformed_triplet_loss"), show_progress=True,
            repeats=repeats)

        # unload values
        del energy_model_no_aug_triplet
        del energy_conf_no_aug_triplet
        del problem_energy_no_aug_triplet
        print(res)
    # %%
    # i found out margin loss can actually make use of dec strat so do it
    if is_image_data:
        energy_model_no_aug_margin_dec_strat = get_network(dataset_info, architecture, num_classes=1).to(device)
        negative_sampling_module_no_aug_margin_dec_strat = BatchNegativeSampler(
            TransformLatentSamplingStrategy(
                transform_sequence=transform_seq, ), transform_true_function
            =transform_true_function, augment_function=affine_augment,
            decision_strategy=dec_strat,
        )

        energy_conf_no_aug_margin_dec_strat = train_or_load_energy_model(
            copy.deepcopy(energy_model_no_aug_margin_dec_strat), model_dir_path,
            f"{modelname}_energy2_only_blur_aug_margin_loss_dec_strat", train_loader,
            val_loader, trainer_kwargs={
                "accelerator": "auto",
                "max_epochs": dataset_info.epochs // 2,
                "precision": "16-mixed" if dataset_info.name not in ["modelnet10"] else "32",
            }, negative_sampling_module=negative_sampling_module_no_aug_margin_dec_strat, load_if_exists=True,
            loss_type="margin", monitor="val_auroc")
        energy_conf_no_aug_margin_dec_strat.to(device).eval()
        problem_energy_no_aug_margin_dec_strat = TransformationProblem(energy_conf_no_aug_margin_dec_strat,
                                                                       transform_seq,
                                                                       consolidate_method="consolidate_simple")

        res = load_or_run_evaluate_confidence_and_search(
            model, optimizer=optimizer, problem=problem_energy_no_aug_margin_dec_strat,
            test_loader=test_loader_transformed, max_batch_override=dataset_info.batch_size_search,
            save_path=savepath("learned_energy_confidence_only_blur_aug_transformed_margin_loss_dec_strat"),
            show_progress=True,
            repeats=repeats)

        del energy_model_no_aug_margin_dec_strat
        del energy_conf_no_aug_margin_dec_strat
        del problem_energy_no_aug_margin_dec_strat
        print(res)

    # %%
    if is_image_data:
        import json
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # Load results for each loss type
        res_bce = json.load(open(savepath("learned_energy_confidence_only_blur_aug_transformed_bce_loss"), "r"))
        res_mse = json.load(open(savepath("learned_energy_confidence_only_blur_aug_transformed_mse_loss"), "r"))
        res_disc = json.load(open(savepath("learned_energy_confidence_only_blur_aug_transformed_disc_loss"), "r"))
        res_margin = json.load(
            open(savepath("learned_energy_confidence_only_blur_aug_transformed_margin_loss_dec_strat"), "r"))

        # Prepare data
        loss_types = ["BCE Loss", "MSE Loss", "Discriminator Loss", "Margin Loss"]
        means = [
            res_bce['accuracy_mean'],
            res_mse['accuracy_mean'],
            res_disc['accuracy_mean'],
            res_margin['accuracy_mean']
        ]
        ses = [
            res_bce.get('accuracy_se', 0),
            res_mse.get('accuracy_se', 0),
            res_disc.get('accuracy_se', 0),
            res_margin.get('accuracy_se', 0)
        ]

        # Calculate y-limits based on SE
        ymin = min([m - s for m, s in zip(means, ses)]) - 0.05
        ymax = max([m + s for m, s in zip(means, ses)]) + 0.05

        # Plot
        plt.figure(figsize=(W, W / 2))
        ax = sns.barplot(x=loss_types, y=means, palette="tab10")
        ax.errorbar(x=np.arange(len(loss_types)), y=means, yerr=ses, fmt='none', c='black', capsize=5)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(ymin, ymax)
        # plt.title("Comparison of Loss Types (Blur Augmentations)")
        plt.savefig(os.path.join(figure_path, "loss_type_comparison_blur_augmentations.pdf"), dpi=300)
        plt.savefig(os.path.join(figure_path, "loss_type_comparison_blur_augmentations.pgf"), dpi=300)
        plt.show()

    # %%

    # %%

    # %%
    from utils.transforms.apply import grid_resample_bilinear

    if dataset == "bigger_mnist" and is_image_data:
        from experiment_thesis.dataset_preperation.get_dataset import get_dataset_info, get_dataset

        dataset_info2 = get_dataset_info(dataset)
        dataset_info2.resample_method = grid_resample_bilinear
        dataset_dict2 = get_dataset(dataset_info2, path=experiment_files_path_data, batch_size=dataset_info2.batch_size)
        transform_name = dataset_info2.transform_seq_name

        dataset_dict2.keys()
        dataset_train_bilinear = dataset_dict2['train_dataset']
        dataset_val_bilinear = dataset_dict2['val_dataset']
        dataset_test_bilinear = dataset_dict2['test_dataset']
        train_loader_bilinear = dataset_dict2['train_loader']
        val_loader_bilinear = dataset_dict2['val_loader']
        test_loader_bilinear = dataset_dict2['test_loader']
        train_loader_transformed_bilinear = dataset_dict2['train_loader_transformed']
        val_loader_transformed_bilinear = dataset_dict2['val_loader_transformed']
        test_loader_transformed_bilinear = dataset_dict2['test_loader_transformed']
        train_loader_no_shuffle_bilinear = dataset_dict2['train_loader_no_shuffle']
        # %%
        energy_model_bilinear = get_network(dataset_info, architecture, num_classes=1).to(device)
        negative_sampling_module_bilinear = BatchNegativeSampler(
            TransformLatentSamplingStrategy(
                transform_sequence=transform_seq, ), transform_true_function
            =transform_true_function, augment_function=affine_augment,
            decision_strategy=dec_strat,
        )
        energy_conf_bilinear = train_or_load_energy_model(
            energy_model_bilinear, model_dir_path, f"{modelname}_energy2_bilinear", train_loader_bilinear,
            val_loader_bilinear, trainer_kwargs={
                "accelerator": "auto",
                "max_epochs": dataset_info.epochs // 2,
                "precision": "16-mixed" if dataset_info.name not in ["modelnet10"] else "32",
            }, negative_sampling_module=negative_sampling_module_bilinear, load_if_exists=True)

        energy_conf_bilinear.to(device).eval()
        problem_energy_bilinear = TransformationProblem(energy_conf_bilinear, transform_seq,
                                                        consolidate_method="consolidate_simple")
        res = load_or_run_evaluate_confidence_and_search(
            model, optimizer=optimizer, problem=problem_energy_bilinear,
            test_loader=test_loader_transformed_bilinear, max_batch_override=dataset_info.batch_size_search,
            save_path=savepath("learned_energy_confidence_bilinear_transformed"), show_progress=True,
            repeats=repeats)

        del energy_model_bilinear
        del energy_conf_bilinear
        del problem_energy_bilinear
        print(res)

        energy_model_bilinear_no_aug = get_network(dataset_info, architecture, num_classes=1).to(device)
        negative_sampling_module_bilinear_no_aug = BatchNegativeSampler(
            TransformLatentSamplingStrategy(
                transform_sequence=transform_seq, ), transform_true_function
            =None, augment_function=None,
            decision_strategy=dec_strat,
        )
        energy_conf_bilinear_no_aug = train_or_load_energy_model(
            energy_model_bilinear_no_aug, model_dir_path, f"{modelname}_energy2_bilinear_no_aug", train_loader_bilinear,
            val_loader_bilinear, trainer_kwargs={
                "accelerator": "auto",
                "max_epochs": dataset_info.epochs // 2,
                "precision": "16-mixed" if dataset_info.name not in ["modelnet10"] else "32",
            }, negative_sampling_module=negative_sampling_module_bilinear_no_aug, load_if_exists=True)

        energy_conf_bilinear_no_aug.to(device).eval()
        problem_energy_bilinear_no_aug = TransformationProblem(energy_conf_bilinear_no_aug, transform_seq,
                                                               consolidate_method="consolidate_simple")
        res = load_or_run_evaluate_confidence_and_search(
            model, optimizer=optimizer, problem=problem_energy_bilinear_no_aug,
            test_loader=test_loader_transformed_bilinear, max_batch_override=dataset_info.batch_size_search,
            save_path=savepath("learned_energy_confidence_bilinear_no_aug_transformed"), show_progress=True,
            repeats=repeats)

        del energy_model_bilinear_no_aug
        del energy_conf_bilinear_no_aug
        del problem_energy_bilinear_no_aug
        print(res)

        # now bilinear with no dec strat
        energy_model_bilinear_no_aug_no_dec_strat = get_network(dataset_info, architecture, num_classes=1).to(device)
        negative_sampling_module_bilinear_no_aug_no_dec_strat = BatchNegativeSampler(
            TransformLatentSamplingStrategy(
                transform_sequence=transform_seq, ), transform_true_function
            =None, augment_function=None,
            decision_strategy=None,
        )
        energy_conf_bilinear_no_aug_no_dec_strat = train_or_load_energy_model(
            energy_model_bilinear_no_aug_no_dec_strat, model_dir_path,
            f"{modelname}_energy2_bilinear_no_aug_no_dec_strat", train_loader_bilinear,
            val_loader_bilinear, trainer_kwargs={
                "accelerator": "auto",
                "max_epochs": dataset_info.epochs // 2,
                "precision": "16-mixed" if dataset_info.name not in ["modelnet10"] else "32",
            }, negative_sampling_module=negative_sampling_module_bilinear_no_aug_no_dec_strat, load_if_exists=True)

        energy_conf_bilinear_no_aug_no_dec_strat.to(device).eval()
        problem_energy_bilinear_no_aug_no_dec_strat = TransformationProblem(energy_conf_bilinear_no_aug_no_dec_strat,
                                                                            transform_seq,
                                                                            consolidate_method="consolidate_simple")
        res = load_or_run_evaluate_confidence_and_search(
            model, optimizer=optimizer, problem=problem_energy_bilinear_no_aug_no_dec_strat,
            test_loader=test_loader_transformed_bilinear, max_batch_override=dataset_info.batch_size_search,
            save_path=savepath("learned_energy_confidence_bilinear_no_aug_no_dec_strat_transformed"),
            show_progress=True,
            repeats=repeats)

        del energy_model_bilinear_no_aug_no_dec_strat
        del energy_conf_bilinear_no_aug_no_dec_strat
        del problem_energy_bilinear_no_aug_no_dec_strat
        print(res)
import torch
import os
import gc

def set_cuda_memory_limits():
    """Prevent PyTorch from falling back to system RAM"""
    if torch.cuda.is_available():
        # Disable fallback to CPU memory
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use max 95% of GPU memory


def clear_gpu_memory():
    """Aggressively clear GPU memory between experiments"""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()



if __name__ == "__main__":
    set_cuda_memory_limits()

    try:
        try:
            run_2_2()
        except Exception as e:
            print(f"Experiment run2")
        finally:
            clear_gpu_memory()
    except Exception:
        print("Experiment interrupted by user.")

    try:
        try:
            run_2_3()
        except Exception as e:
            print(f"Experiment run2")
        finally:
            clear_gpu_memory()
    except Exception:
        print("Experiment interrupted by user.")

    try:
        try:
            run_a_opt_hyper()
        except Exception as e:
            print(f"Experiment run2")
        finally:
            clear_gpu_memory()
    except Exception:
        print("Experiment interrupted by user.")

    try:
        try:
            run_a_opt_hyper_2()
        except Exception as e:
            print(f"Experiment run2")
        finally:
            clear_gpu_memory()
    except Exception:
        print("Experiment interrupted by user.")

    try:
        try:
            run_3_1()
        except Exception as e:
            print(f"Experiment run2")
        finally:
            clear_gpu_memory()
    except Exception:
        print("Experiment interrupted by user.")

    try:
        try:
            run_3_3()
        except Exception as e:
            print(f"Experiment run2")
        finally:
            clear_gpu_memory()
    except Exception:
        print("Experiment interrupted by user.")


    exit()



