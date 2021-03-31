
import atexit
import logging
import hydra
import functools
import pdb
import numpy as np
import os
import scipy.stats
import skimage.io

import precog.utils.log_util as logu
import precog.utils.tfutil as tfutil
import precog.interface as interface
import precog.plotting.plot as plot
import utility as util

log = logging.getLogger(os.path.basename(__file__))

@hydra.main(config_path='conf/esp_infer_config.yaml')
def main(cfg):
    assert(cfg.main.plot or cfg.main.compute_metrics)
    output_directory = os.path.realpath(os.getcwd())
    images_directory = os.path.join(output_directory, 'outputs')
    os.mkdir(images_directory)

    atexit.register(logu.query_purge_directory, output_directory)

    # Instantiate the session.
    sess = tfutil.create_session(
            allow_growth=cfg.hardware.allow_growth,
            per_process_gpu_memory_fraction=cfg.hardware.per_process_gpu_memory_fraction)

    # Load the model and the tensor collections.
    log.info("Loading the model...")
    ckpt, graph, tensor_collections = tfutil.load_annotated_model(cfg.model.directory, sess)
    inference = interface.ESPInference(tensor_collections)
    sample_metrics = tfutil.get_collection_dict(tensor_collections['sample_metric'])
    if cfg.main.compute_metrics:
        infer_metrics = tfutil.get_collection_dict(tensor_collections['infer_metric'])
        metrics = {**infer_metrics, **sample_metrics}
        all_metrics = {_: [] for _ in metrics.keys()}

    # Instantiate the dataset.
    cfg.dataset.params.T = inference.metadata.T
    cfg.dataset.params.B = inference.metadata.B
    cfg.dataset.params.A = inference.metadata.A
    cfg.dataset.params.W = inference.phi_metadata.H
    log.info("\n\nConfig:\n===\n{}".format(cfg.pretty()))
    dataset = hydra.utils.instantiate(cfg.dataset, **cfg.dataset.params)

    payloads = {}

    log.info("Beginning evaluation. Model: {}".format(ckpt))    
    for epoch in range(cfg.epochs):
        dataset.reset_split(cfg.split)
        count_images = 0
        count_json = 0
        while True:
            minibatch = dataset.get_minibatch(
                    split=cfg.split,
                    input_singleton=inference.training_input,
                    is_training=False)
            if minibatch is None:
                break
            sessrun = functools.partial(sess.run, feed_dict=minibatch)
            # Run sampling and convert to numpy.
            sampled_output = inference.sampled_output.to_numpy(sessrun)
            experts = inference.training_input.experts.to_numpy(sessrun)
            
            if cfg.main.compute_metrics:
                # Get experts in numpy version.
                # Compute and store metrics.
                metrics_results = dict(zip(metrics.keys(), sessrun(list(metrics.values()))))            
                for k, val in metrics_results.items():
                    all_metrics[k].append(val)

            if cfg.main.plot and epoch == 0:
                for b in range(inference.metadata.B):
                    im = plot.plot_sample(
                            sampled_output, experts,
                            b=b, partial_write_np_image_to_tb=lambda x: x,
                            figsize=cfg.images.figsize,
                            bev_kwargs=cfg.plotting.bev_kwargs)
                    save_path = "{}/sample{:05d}_epoch{}.png".format(
                            images_directory, count_images, epoch)
                    skimage.io.imsave(save_path, im[0,...,:3])
                    count_images += 1
            
            # save trajectory data
            # forecasts (B, K, A, T, D) => (B, A, K, T, D)
            all_esp_coords = sampled_output.rollout.S_grid_frame.swapaxes(1,2)
            # ground_truth (B, A, T, D)
            all_expert_coords = experts.S_future_grid_frame
            # pasts (B, A, Tpast, D)
            all_past_coords = sampled_output.phi.S_past_grid_frame
            for b in range(inference.metadata.B):
                if count_json in payloads:
                    payload = payloads[count_json]
                    payload['player_forecast'] = np.concatenate((
                            payload['player_forecast'],
                            all_esp_coords[b, 0]), axis=0)
                    payload['agent_forecasts'] = np.concatenate((
                            payload['agent_forecasts'],
                            all_esp_coords[b, 1:]), axis=1)
                else:
                    payload = {
                        'player_past': all_past_coords[b, 0],
                        'agent_pasts': all_past_coords[b, 1:],
                        'player_expert': all_expert_coords[b, 0],
                        'agent_experts': all_expert_coords[b, 1:],
                        'player_forecast': all_esp_coords[b, 0],
                        'agent_forecasts': all_esp_coords[b, 1:],
                        'overhead_features': sampled_output.phi.overhead_features[b],
                    }
                    payloads[count_json] = payload
                count_json += 1

            # debugging
            # print("all_esp_coords.shape", all_esp_coords.shape)
            # print("all_expert_coords.shape", all_expert_coords.shape)
            # print("all_past_coords.shape", all_past_coords.shape)
            # raise NotImplementedError()

        if cfg.main.compute_metrics:
            print(f"Final metrics for epoch {epoch}:")
            for k, vals in all_metrics.items():
                print("Mean,sem '{}'={:.3f} +- {:.3f}".format(
                        k, np.mean(vals), scipy.stats.sem(vals, axis=None)))
    
    for idx, payload in payloads.items():
        filename = "sample{:05d}".format(idx)
        util.save_datum(payload, images_directory, filename)


if __name__ == '__main__':
    main()
