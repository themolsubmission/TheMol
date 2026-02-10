import torch
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss

import torch.nn as nn


prop_mean = torch.tensor([3.87390190e+02, 3.31457479e+00, 1.04145914e+02, 1.27685250e+00,
        4.97280500e+00, 7.60752555e+01, 1.60859374e+02, 5.33976750e+00,
        3.39176455e-01, 2.70027870e+01, 2.30398450e+00, 8.62721000e-01,
        1.44126350e+00, 7.15643150e+00, 5.87215218e-01, 4.17450000e-02])

prop_std = torch.tensor([120.9011186 ,   1.77640817,  31.5745483 ,   1.44678132,
          2.26865763,  42.02033085,  48.83558139,   3.44042127,
          0.19763768,   8.35079423,   1.09886119,   0.86003458,
          0.95412579,   3.06503437,   0.20763273,   0.20000589])


@register_loss("unimol_MAE")
class UniMolOptimalLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        self.seed = task.seed
        self.dist_mean = 6.312581655060595
        self.dist_std = 3.3899264663911888
        self.idx = 0
        #self.bond_extractor = CompleteBondExtractor(task)
        self.bond_extractor = CompleteBondExtractor(task)
        self.task = task
    def forward(self, model, sample, reduce=True):
        self.idx += 1
        input_key = "net_input"
        target_key = "target"
        noised_tokens = sample[target_key]["tokens_target"].ne(self.padding_idx)

        
        #masked_tokens = None
        pair_padding_mask = sample[input_key]['src_distance'].ne(self.padding_idx)
        
        """At Generative Stage Only"""
        if self.task.args.task_sub == "generative":
            sample[input_key]['src_distance'] = sample[target_key]['distance_target']
            sample[input_key]['src_tokens'][sample[target_key]["tokens_target"] != 0] = sample[target_key]["tokens_target"][sample[target_key]["tokens_target"] != 0]
            sample[input_key]['src_coord'] = sample[target_key]['coord_target']
        
            
        sample[input_key]['src_tokens'][sample[input_key]['src_tokens'] == 1] = 0 
        sample[input_key]['src_tokens'][sample[input_key]['src_tokens'] == 2] = 0         
        sample = apply_random_rotation_to_target_coords(sample, target_key, apply_prob=0.7)
        
        
        padding_mask = sample[input_key]['src_tokens'].ne(self.padding_idx)
        
        masked_tokens = padding_mask

        bond_types = self.bond_extractor.extract_bond_types_from_sample(sample)
        sample_size = masked_tokens.long().sum()
        sample[input_key]['src_bond_type'] = bond_types
        
        
        # ae_only, flow_only, dual
        
        #mode = "dual"
        if self.task.args.training == "flow_only":
            mode = "flow_only"
        elif self.task.args.training == "ae_only":
            mode = "ae_only"
        else:
            mode = "dual"
        

        (
            logits_encoder,
            encoder_distance,
            encoder_coord,
            encoder_x_norm,
            decoder_x_norm,
            delta_encoder_pair_rep_norm,
            delta_decoder_pair_rep_norm,
            (z,q_z_given_x,p_z,latent_emb,std),
            flow_loss_dict
        ) = model(**sample[input_key], mode=mode, encoder_masked_tokens=None)
        
        # sample[input_key]['src_distance'] = sample[target_key]['distance_target']
        # sample[input_key]['src_tokens'][sample[target_key]["tokens_target"] != 0] = sample[target_key]["tokens_target"][sample[target_key]["tokens_target"] != 0]
        # sample[input_key]['src_coord'] = sample[target_key]['coord_target']
        token_label = sample[input_key]['src_tokens'].clone()
        token_label[sample[target_key]["tokens_target"] != 0] = sample[target_key]["tokens_target"][sample[target_key]["tokens_target"] != 0]
        weight_re = (1 - (self.idx % 73584)) / 73584
        weight_re = 0.
        #target = sample[target_key]["tokens_target"]
        if masked_tokens is not None:
            target = sample[input_key]['src_tokens'][masked_tokens]
            target = token_label[masked_tokens]
        token_loss = F.nll_loss(
            F.log_softmax(logits_encoder[padding_mask], dim=-1, dtype=torch.float32),
            target,
            ignore_index=self.padding_idx,
            reduction="mean",
        )
        
        noised_token_acc = (logits_encoder[noised_tokens].argmax(dim=-1) == token_label[noised_tokens]).long().sum()/noised_tokens.sum()
        masked_pred = logits_encoder[padding_mask].argmax(dim=-1)
        masked_hit = (masked_pred == target).long().sum()
        masked_cnt = sample_size
        loss = token_loss * 0.5 * self.args.masked_token_loss
        logging_output = {
            "sample_size": 1,
            "bsz": sample[target_key]["tokens_target"].size(0),
            "seq_len": sample[target_key]["tokens_target"].size(1)
            * sample[target_key]["tokens_target"].size(0),
            "token_loss": token_loss.data,
            "masked_token_hit": masked_hit.data,
            "masked_token_cnt": masked_cnt,
            #"latent_mean":latent_emb[masked_tokens].mean().data,
            #"latent_mean":torch.mean((torch.abs(latent_emb[masked_tokens]) < 0.2).float()).data,
            "latent_mean":torch.mean(torch.abs(latent_emb[masked_tokens])).data,
            #"latent_std":torch.mean()
            "noised_acc" : noised_token_acc.data,
            
            "latent_std":torch.mean((torch.abs(std[masked_tokens]) > 0.7).float()).data

            # "latent_mean":latent_emb.mean().data,
            # "latent_std":std.mean().data
        }
        
        """OT-CFM"""
        # #flow_loss = None
        # if flow_loss is not None:
        #     #flow_loss = flow_loss[masked_tokens].mean()
        #     loss = loss + flow_loss[masked_tokens].mean() 
        #     logging_output["flow_loss"] = flow_loss[masked_tokens].mean().data
        """FrameFlow"""
        if mode == "ae_only":
            flow_loss_dict = None
        
        if mode in ["flow_only", "dual"]:
            annealing_steps = 50000
            flow_weight = min(10.0, model.get_num_updates() / annealing_steps)
            #flow_weight = 10.
            flow_loss = flow_loss_dict['loss']
            loss = loss + flow_loss * flow_weight
            logging_output["flow_loss"] = flow_loss.data
            
            if "f_loss t=[0,25)" in flow_loss_dict.keys():
                logging_output["f_loss[0,25)"] = flow_loss_dict['f_loss t=[0,25)']
            else:
                logging_output["f_loss[0,25)"] = 0.
                
            if "f_loss t=[25,50)" in flow_loss_dict.keys():
                logging_output["f_loss[25,50)"] = flow_loss_dict['f_loss t=[25,50)']
            else:
                logging_output["f_loss[25,50)"] = 0.
                
            if "f_loss t=[50,75)" in flow_loss_dict.keys():
                logging_output["f_loss[50,75)"] = flow_loss_dict['f_loss t=[50,75)']
            else:
                logging_output["f_loss[50,75)"] = 0.
                
            if "f_loss t=[75,100)" in flow_loss_dict.keys():
                logging_output["f_loss[75,100)"] = flow_loss_dict['f_loss t=[75,100)']
            else:
                logging_output["f_loss[75,100)"] = 0.

        if len(encoder_coord)==2:

            rot_mat, pred_coord = encoder_coord
            coord_target = sample[target_key]['coord_target']
            coord_target = torch.bmm(coord_target, rot_mat)
            
            coord_loss = F.smooth_l1_loss(
                pred_coord[masked_tokens].view(-1, 3).float(),
                coord_target[masked_tokens].view(-1, 3),
                reduction="mean",
                beta=1.0,
            )
            coord_loss_log_version = F.l1_loss(
                pred_coord[masked_tokens].view(-1, 3).float(),
                coord_target[masked_tokens].view(-1, 3),
                reduction="mean",
            )
            loss = loss + coord_loss * self.args.masked_coord_loss * 10
            # restore the scale of loss for displaying
            logging_output["coord_loss"] = coord_loss_log_version.data
            
            
        else:
            coord_target = sample[target_key]['coord_target']
            coord_loss = F.smooth_l1_loss(
                encoder_coord[masked_tokens].view(-1, 3).float(),
                coord_target[masked_tokens].view(-1, 3),
                reduction="mean",
                beta=1.0,
            )
            coord_loss_log_version = F.l1_loss(
                encoder_coord[masked_tokens].view(-1, 3).float(),
                coord_target[masked_tokens].view(-1, 3),
                reduction="mean",
            )
            loss = loss + coord_loss * self.args.masked_coord_loss * 10
            # restore the scale of loss for displaying
            logging_output["coord_loss"] = coord_loss_log_version.data
        #loss = weight_re
        if encoder_distance is not None:
            dist_masked_tokens = masked_tokens
            masked_dist_loss, masked_dist_loss_log = self.cal_dist_loss_v2(
                sample, encoder_distance, dist_masked_tokens, target_key, normalize=True
            )
            loss = loss + masked_dist_loss * 10 * self.args.masked_dist_loss #+ gram_loss * 10
            logging_output["masked_dist_loss"] = masked_dist_loss_log.data
            #logging_output["gram_loss"] = gram_loss.data
        
            
        if self.args.encoder_x_norm_loss > 0 and encoder_x_norm is not None:
            loss = loss + self.args.encoder_x_norm_loss * encoder_x_norm
            logging_output["encoder_x_norm_loss"] = encoder_x_norm.data

        if z is not None:            
            
            """Default with Debug"""
            kl_div_element_wise = torch.distributions.kl.kl_divergence(q_z_given_x, p_z)
            
            kl_per_token = kl_div_element_wise.sum(dim=-1)
            
            #kl_divs = kl_div_element_wise[padding_mask]
            #kl_divs = kl_div_element_wise
            kl_loss = kl_per_token[padding_mask].mean()
            annealing_steps = 100000
            kl_weight = min(1.0, model.get_num_updates() / annealing_steps)
            #kl_weight = 1
            logging_output["KL_loss"] = kl_loss.data
            loss = loss + (kl_loss * kl_weight) * 0.01
            
        if (
            self.args.encoder_delta_pair_repr_norm_loss > 0
            and delta_encoder_pair_rep_norm is not None
        ):
            loss = (
                loss + self.args.encoder_delta_pair_repr_norm_loss * delta_encoder_pair_rep_norm
            )
            logging_output[
                "encoder_delta_pair_repr_norm_loss"
            ] = delta_encoder_pair_rep_norm.data

        if self.args.decoder_x_norm_loss > 0 and decoder_x_norm is not None:
            loss = loss + self.args.decoder_x_norm_loss * decoder_x_norm
            logging_output["decoder_x_norm_loss"] = decoder_x_norm.data

        if (
            self.args.decoder_delta_pair_repr_norm_loss > 0
            and delta_decoder_pair_rep_norm is not None
        ):
            loss = (
                loss + self.args.decoder_delta_pair_repr_norm_loss * delta_decoder_pair_rep_norm
            )
            logging_output[
                "decoder_delta_pair_repr_norm_loss"
            ] = delta_decoder_pair_rep_norm.data

        #logging_output["loss"] = loss.data
        
        """OT-CFM"""
        # logging_output["loss"] = flow_loss[masked_tokens].mean() 
        # loss = flow_loss[masked_tokens].mean() 
        
        if mode in ["ae_only", "dual"]:
            logging_output["loss"] = loss.data
        else:
            """FrameFlow"""
            logging_output["loss"] = flow_loss
            loss = flow_loss
            
                    
        return loss, 1, logging_output

    
    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        #metrics.log_scalar("seq_len", seq_len / bsz, 1, round=3)
        
        masked_loss = sum(log.get("token_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "token_loss", masked_loss / sample_size, sample_size, round=3
        )

        masked_acc = sum(
            log.get("masked_token_hit", 0) for log in logging_outputs
        ) / sum(log.get("masked_token_cnt", 0) for log in logging_outputs)
        metrics.log_scalar("masked_acc", masked_acc, sample_size, round=3)

        noised_token_loss = sum(log.get("noised_acc", 0) for log in logging_outputs)
        metrics.log_scalar(
            "noised_acc", noised_token_loss / sample_size, sample_size, round=3
        )
        
        
        coord_loss = sum(
            log.get("coord_loss", 0) for log in logging_outputs
        )
        if coord_loss > 0:
            metrics.log_scalar(
                "coord_loss",
                coord_loss / sample_size,
                sample_size,
                round=4,
            )
        lm = sum(log.get("latent_mean", 0) for log in logging_outputs)
        ls = sum(log.get("latent_std", 0) for log in logging_outputs)
        metrics.log_scalar(
            "lat_mean", lm/sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "lat_std", ls/sample_size, sample_size, round=2
        )
                
        masked_dist_loss = sum(
            log.get("masked_dist_loss", 0) for log in logging_outputs
        )
        if masked_dist_loss > 0:
            metrics.log_scalar(
                "masked_dist_loss", masked_dist_loss / sample_size, sample_size, round=3
            )

        # gram_loss = sum(
        #     log.get("gram_loss", 0) for log in logging_outputs
        # )
        # if gram_loss > 0:
        #     metrics.log_scalar(
        #         "gram_loss", gram_loss / sample_size, sample_size, round=3
        #     )
            
            
        flow_loss = sum(log.get("flow_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "Flow_loss", flow_loss/sample_size, sample_size, round=5
        )

        f_1 = sum(log.get("f_loss[0,25)", 0) for log in logging_outputs)
        f_2 = sum(log.get("f_loss[25,50)", 0) for log in logging_outputs)
        f_3 = sum(log.get("f_loss[50,75)", 0) for log in logging_outputs)
        f_4 = sum(log.get("f_loss[75,100)", 0) for log in logging_outputs)

        metrics.log_scalar(
            "1st_Bin", f_1/sample_size, sample_size, round=2
        )
        metrics.log_scalar(
            "2nd_Bin", f_2/sample_size, sample_size, round=2
        )
        metrics.log_scalar(
            "3rd_Bin", f_3/sample_size, sample_size, round=2
        )
        metrics.log_scalar(
            "4th_Bin", f_4/sample_size, sample_size, round=2
        )        

                
        kl_loss = sum(log.get("KL_loss", 0) for log in logging_outputs)
        
        #if encoder_x_norm_loss > 0:
        metrics.log_scalar(
            "KL_loss", kl_loss / sample_size, sample_size, round=3
        )
        
        encoder_x_norm_loss = sum(log.get("encoder_x_norm_loss", 0) for log in logging_outputs)
        if encoder_x_norm_loss > 0:
            metrics.log_scalar(
                "encoder_x_norm_loss", encoder_x_norm_loss / sample_size, sample_size, round=3
            )

            
        encoder_delta_pair_repr_norm_loss = sum(
            log.get("encoder_delta_pair_repr_norm_loss", 0) for log in logging_outputs
        )
        if encoder_delta_pair_repr_norm_loss > 0:
            metrics.log_scalar(
                "encoder_delta_pair_repr_norm_loss",
                encoder_delta_pair_repr_norm_loss / sample_size,
                sample_size,
                round=3,
            )

        decoder_x_norm_loss = sum(log.get("decoder_x_norm_loss", 0) for log in logging_outputs)
        if decoder_x_norm_loss > 0:
            metrics.log_scalar(
                "decoder_x_norm_loss", decoder_x_norm_loss / sample_size, sample_size, round=3
            )

        decoder_delta_pair_repr_norm_loss = sum(
            log.get("decoder_delta_pair_repr_norm_loss", 0) for log in logging_outputs
        )
        if decoder_delta_pair_repr_norm_loss > 0:
            metrics.log_scalar(
                "decoder_delta_pair_repr_norm_loss",
                decoder_delta_pair_repr_norm_loss / sample_size,
                sample_size,
                round=3,
            )
        
        metrics.log_scalar("seq_len", seq_len / bsz, 1, round=3)
    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def cal_dist_loss(self, sample, dist, masked_tokens, target_key, normalize=False):
        dist_masked_tokens = masked_tokens
        masked_distance = dist[dist_masked_tokens, :]
        masked_distance_target = sample[target_key]["distance_target"][
            dist_masked_tokens
        ]
        non_pad_pos = masked_distance_target > 0
        if normalize:
            masked_distance_target = (
                masked_distance_target.float() - self.dist_mean
            ) / self.dist_std
        masked_dist_loss = F.smooth_l1_loss(
            masked_distance[non_pad_pos].view(-1).float(),
            masked_distance_target[non_pad_pos].view(-1),
            reduction="mean",
            beta=1.0,
        )
        return masked_dist_loss
    
    def cal_dist_loss_v2(self, sample, dist, masked_tokens, target_key, normalize=False):
        dist.diagonal(offset=0, dim1=-2, dim2=-1).fill_(0)
        dist_masked_tokens = masked_tokens.unsqueeze(-1) & masked_tokens.unsqueeze(-2)
        masked_distance = dist[dist_masked_tokens]
        masked_distance_target = sample['target']["distance_target"][
            dist_masked_tokens
        ]
        #non_pad_pos = masked_distance_target > 0
        # if normalize:
        #     masked_distance_target = (
        #         masked_distance_target.float() - self.dist_mean
        #     ) / self.dist_std
        masked_dist_loss = F.smooth_l1_loss(
            masked_distance.view(-1).float(),
            masked_distance_target.view(-1),
            reduction="mean",
            beta=1.0,
        )

        masked_dist_loss_log = F.l1_loss(
            masked_distance.view(-1).float(),
            masked_distance_target.view(-1),
            reduction="mean"
        )
                
        dist = dist * dist_masked_tokens
        #gram_loss = EDMLoss()(dist[:,1:,1:])
        return masked_dist_loss, masked_dist_loss_log


import pickle
@register_loss("unimol_MAE2")
class UniMolOptimalLoss2(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        self.seed = task.seed
        self.dist_mean = 6.312581655060595
        self.dist_std = 3.3899264663911888
        self.idx = 0
        self.bond_extractor = CompleteBondExtractor(task)
        #self.bond_extractor = FixedCompleteBondExtractor(task)
        # Load property table if available
        import os
        property_table_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'total_dict.pickle')
        with open(property_table_path, 'rb') as f:
            self.property_table = pickle.load(f)
        self.task = task

    def compute_rotation_consistency_loss(
        self,
        model,
        original_coord,
        src_tokens,
        src_distance,
        src_edge_type,
        src_bond_type,
        rot_mat_rotated,
        R_random
    ):
        """
        Compute rotation consistency loss.

        For the same molecule:
        - Original coordinates → rot_mat_1 → canonical
        - Rotated coordinates → rot_mat_2 → canonical (should be the same)

        Mathematical relationship: rot_mat_1 = R_random @ rot_mat_2
        Therefore: R_random^T @ rot_mat_1 ≈ rot_mat_2

        Args:
            model: Model instance
            original_coord: Original coordinates before rotation [B, N, 3]
            src_tokens: Atom tokens [B, N]
            src_distance: Distance matrix [B, N, N]
            src_edge_type: Edge type [B, N, N]
            src_bond_type: Bond type [B, N, N]
            rot_mat_rotated: Predicted rot_mat from rotated coordinates (rot_mat_2) [B, 3, 3]
            R_random: Applied random rotation matrix [B, 3, 3]

        Returns:
            rotation_consistency_loss: Scalar loss value
        """
        # Predict rot_mat from original coordinates (rot_mat_1)
        _, rot_mat_original = model.rot(
            src_tokens,
            src_distance,
            original_coord,
            src_edge_type,
            src_bond_type,
        )

        # Mathematical relationship: rot_mat_2 = R_random @ rot_mat_1
        expected_rot_mat_2 = torch.bmm(R_random, rot_mat_original)

        # Frobenius norm loss
        rot_loss = ((rot_mat_rotated - expected_rot_mat_2) ** 2).sum(dim=(-1, -2)).mean()

        return rot_loss

    def forward(self, model, sample, reduce=True):
        self.idx += 1
        input_key = "net_input"
        target_key = "target"
        noised_tokens = sample[target_key]["tokens_target"].ne(self.padding_idx)

        #masked_tokens = None
        pair_padding_mask = sample[input_key]['src_distance'].ne(self.padding_idx)

        """At Generative Stage Only"""
        if self.task.args.task_sub == "generative":
            sample[input_key]['src_distance'] = sample[target_key]['distance_target']
            sample[input_key]['src_tokens'][sample[target_key]["tokens_target"] != 0] = sample[target_key]["tokens_target"][sample[target_key]["tokens_target"] != 0]
            sample[input_key]['src_coord'] = sample[target_key]['coord_target']

        sample[input_key]['src_tokens'][sample[input_key]['src_tokens'] == 1] = 0
        sample[input_key]['src_tokens'][sample[input_key]['src_tokens'] == 2] = 0

        # Save original coordinates (for rotation consistency loss computation)
        original_coord = sample[input_key]['src_coord'].clone()

        sample, R_random, rotation_applied = apply_random_rotation_to_target_coords(sample, target_key, apply_prob=1.0)

        padding_mask = sample[input_key]['src_tokens'].ne(self.padding_idx)

        masked_tokens = padding_mask

        bond_types = self.bond_extractor.extract_bond_types_from_sample(sample)
        sample_size = masked_tokens.long().sum()
        sample[input_key]['src_bond_type'] = bond_types

        # ae_only, flow_only, dual
        #mode = "dual"
        if self.task.args.training == "flow_only":
            mode = "flow_only"
        elif self.task.args.training == "ae_only":
            mode = "ae_only"
        else:
            mode = "dual"

        (
            logits_encoder,
            encoder_distance,
            encoder_coord,
            encoder_x_norm,
            decoder_x_norm,
            delta_encoder_pair_rep_norm,
            delta_decoder_pair_rep_norm,
            (z,q_z_given_x,p_z,latent_emb,std),
            flow_loss_dict,
            prop_pred
        ) = model(**sample[input_key], mode=mode, encoder_masked_tokens=None)

        # sample[input_key]['src_distance'] = sample[target_key]['distance_target']
        # sample[input_key]['src_tokens'][sample[target_key]["tokens_target"] != 0] = sample[target_key]["tokens_target"][sample[target_key]["tokens_target"] != 0]
        # sample[input_key]['src_coord'] = sample[target_key]['coord_target']
        token_label = sample[input_key]['src_tokens'].clone()
        token_label[sample[target_key]["tokens_target"] != 0] = sample[target_key]["tokens_target"][sample[target_key]["tokens_target"] != 0]
        weight_re = (1 - (self.idx % 73584)) / 73584
        weight_re = 0.
        #target = sample[target_key]["tokens_target"]
        if masked_tokens is not None:
            target = sample[input_key]['src_tokens'][masked_tokens]
            target = token_label[masked_tokens]
        token_loss = F.nll_loss(
            F.log_softmax(logits_encoder[padding_mask], dim=-1, dtype=torch.float32),
            target,
            ignore_index=self.padding_idx,
            reduction="mean",
        )
        
        noised_token_acc = (logits_encoder[noised_tokens].argmax(dim=-1) == token_label[noised_tokens]).long().sum()/noised_tokens.sum()
        masked_pred = logits_encoder[padding_mask].argmax(dim=-1)
        masked_hit = (masked_pred == target).long().sum()
        masked_cnt = sample_size
        loss = token_loss * 0.5 * self.args.masked_token_loss
        logging_output = {
            "sample_size": 1,
            "bsz": sample[target_key]["tokens_target"].size(0),
            "seq_len": sample[target_key]["tokens_target"].size(1)
            * sample[target_key]["tokens_target"].size(0),
            "token_loss": token_loss.data,
            "masked_token_hit": masked_hit.data,
            "masked_token_cnt": masked_cnt,
            #"latent_mean":latent_emb[masked_tokens].mean().data,
            #"latent_mean":torch.mean((torch.abs(latent_emb[masked_tokens]) < 0.2).float()).data,
            "latent_mean":torch.mean(torch.abs(latent_emb[masked_tokens])).data,
            #"latent_std":torch.mean()
            "noised_acc" : noised_token_acc.data,
            
            "latent_std":torch.mean((torch.abs(std[masked_tokens]) > 0.7).float()).data

            # "latent_mean":latent_emb.mean().data,
            # "latent_std":std.mean().data
        }
        
        """OT-CFM"""
        # #flow_loss = None
        # if flow_loss is not None:
        #     #flow_loss = flow_loss[masked_tokens].mean()
        #     loss = loss + flow_loss[masked_tokens].mean() 
        #     logging_output["flow_loss"] = flow_loss[masked_tokens].mean().data
        """FrameFlow"""
        if mode == "ae_only":
            flow_loss_dict = None
        
        if mode in ["flow_only", "dual"]:
            annealing_steps = 50000
            flow_weight = min(10.0, model.get_num_updates() / annealing_steps)
            #flow_weight = 10.
            flow_loss = flow_loss_dict['loss']
            loss = loss + flow_loss * flow_weight
            logging_output["flow_loss"] = flow_loss.data
            
            if "f_loss t=[0,25)" in flow_loss_dict.keys():
                logging_output["f_loss[0,25)"] = flow_loss_dict['f_loss t=[0,25)']
            else:
                logging_output["f_loss[0,25)"] = 0.
                
            if "f_loss t=[25,50)" in flow_loss_dict.keys():
                logging_output["f_loss[25,50)"] = flow_loss_dict['f_loss t=[25,50)']
            else:
                logging_output["f_loss[25,50)"] = 0.
                
            if "f_loss t=[50,75)" in flow_loss_dict.keys():
                logging_output["f_loss[50,75)"] = flow_loss_dict['f_loss t=[50,75)']
            else:
                logging_output["f_loss[50,75)"] = 0.
                
            if "f_loss t=[75,100)" in flow_loss_dict.keys():
                logging_output["f_loss[75,100)"] = flow_loss_dict['f_loss t=[75,100)']
            else:
                logging_output["f_loss[75,100)"] = 0.

        if len(encoder_coord)==2:

            rot_mat, pred_coord = encoder_coord
            if len(pred_coord) == 2:
                pred_coord_, pred_bond = pred_coord
                coord_target = sample[target_key]['coord_target']
                coord_target = torch.bmm(coord_target, rot_mat)
                
                coord_loss = F.smooth_l1_loss(
                    pred_coord_[masked_tokens].view(-1, 3).float(),
                    coord_target[masked_tokens].view(-1, 3),
                    reduction="mean",
                    beta=1.0,
                )
                coord_loss_log_version = F.l1_loss(
                    pred_coord_[masked_tokens].view(-1, 3).float(),
                    coord_target[masked_tokens].view(-1, 3),
                    reduction="mean",
                )
                loss = loss + coord_loss * self.args.masked_coord_loss * 10
                # restore the scale of loss for displaying
                logging_output["coord_loss"] = coord_loss_log_version.data
                
                
                dist_masked_tokens = masked_tokens.unsqueeze(-1) & masked_tokens.unsqueeze(-2)
               
                bond_weight = torch.tensor([1,100,200,200,100], device=bond_types.device).float()
                bond_criterion = nn.CrossEntropyLoss(weight=bond_weight, reduction='mean')
                b_loss = bond_criterion(pred_bond[dist_masked_tokens], bond_types[dist_masked_tokens])
                
                loss = loss + b_loss
                logging_output["bond_loss"] = b_loss.data
                bond_acc = (pred_bond[dist_masked_tokens].argmax(-1) == bond_types[dist_masked_tokens]).long().sum()/dist_masked_tokens.sum()
                #bond_acc = (pred_bond[dist_masked_tokens].argmax(-1) == bond_types[dist_masked_tokens]).long().sum()/dist_masked_tokens.sum()
                is_bond = bond_types[dist_masked_tokens] != 0
                isbond_acc = (pred_bond[dist_masked_tokens].argmax(-1)[is_bond] == bond_types[dist_masked_tokens][is_bond]).long().sum()/is_bond.sum()
                
                isnot_bond = bond_types[dist_masked_tokens] == 0
                isnotbond_acc = (pred_bond[dist_masked_tokens].argmax(-1)[isnot_bond] == bond_types[dist_masked_tokens][isnot_bond]).long().sum()/isnot_bond.sum()
                
                logging_output['bond_acc'] = bond_acc.data
                logging_output['Isbond_acc'] = isbond_acc.data
                logging_output['Notbond_acc'] = isnotbond_acc.data

            elif len(pred_coord) == 3:
                pred_coord_refine, pred_bond, pred_coord_ = pred_coord
                coord_target = sample[target_key]['coord_target']
                coord_target = torch.bmm(coord_target, rot_mat)
                
                coord_loss = F.smooth_l1_loss(
                    pred_coord_[masked_tokens].view(-1, 3).float(),
                    coord_target[masked_tokens].view(-1, 3),
                    reduction="mean",
                    beta=1.0,
                )
                coord_loss_log_version = F.l1_loss(
                    pred_coord_[masked_tokens].view(-1, 3).float(),
                    coord_target[masked_tokens].view(-1, 3),
                    reduction="mean",
                )
                loss = loss + coord_loss * self.args.masked_coord_loss * 10
                # restore the scale of loss for displaying
                logging_output["coord_loss"] = coord_loss_log_version.data
                
                coord_refine_loss = F.smooth_l1_loss(
                    pred_coord_refine[masked_tokens].view(-1, 3).float(),
                    coord_target[masked_tokens].view(-1, 3),
                    reduction="mean",
                    beta=1.0,
                )
                coord_refine_loss_log_version = F.l1_loss(
                    pred_coord_refine[masked_tokens].view(-1, 3).float(),
                    coord_target[masked_tokens].view(-1, 3),
                    reduction="mean",
                )
                loss = loss + coord_loss * self.args.masked_coord_loss * 10 + coord_refine_loss * self.args.masked_coord_loss * 10
                
                # restore the scale of loss for displaying
                logging_output["coord_loss"] = coord_loss_log_version.data
                logging_output["coord_refine_loss"] = coord_refine_loss_log_version.data
                                
                dist_masked_tokens = masked_tokens.unsqueeze(-1) & masked_tokens.unsqueeze(-2)
               
                bond_weight = torch.tensor([1,100,200,200,100], device=bond_types.device).float()
                bond_criterion = nn.CrossEntropyLoss(weight=bond_weight, reduction='mean')
                b_loss = bond_criterion(pred_bond[dist_masked_tokens], bond_types[dist_masked_tokens])
                
                loss = loss + b_loss
                logging_output["bond_loss"] = b_loss.data
                bond_acc = (pred_bond[dist_masked_tokens].argmax(-1) == bond_types[dist_masked_tokens]).long().sum()/dist_masked_tokens.sum()
                #bond_acc = (pred_bond[dist_masked_tokens].argmax(-1) == bond_types[dist_masked_tokens]).long().sum()/dist_masked_tokens.sum()
                is_bond = bond_types[dist_masked_tokens] != 0
                isbond_acc = (pred_bond[dist_masked_tokens].argmax(-1)[is_bond] == bond_types[dist_masked_tokens][is_bond]).long().sum()/is_bond.sum()
                
                isnot_bond = bond_types[dist_masked_tokens] == 0
                isnotbond_acc = (pred_bond[dist_masked_tokens].argmax(-1)[isnot_bond] == bond_types[dist_masked_tokens][isnot_bond]).long().sum()/isnot_bond.sum()
                
                logging_output['bond_acc'] = bond_acc.data
                logging_output['Isbond_acc'] = isbond_acc.data
                logging_output['Notbond_acc'] = isnotbond_acc.data
                
                if rotation_applied:
                    rot_consistency_loss = self.compute_rotation_consistency_loss(
                        model=model,
                        original_coord=original_coord,
                        src_tokens=sample[input_key]['src_tokens'],
                        src_distance=sample[input_key]['src_distance'],
                        src_edge_type=sample[input_key]['src_edge_type'],
                        src_bond_type=sample[input_key]['src_bond_type'],
                        rot_mat_rotated=rot_mat,
                        R_random=R_random
                    )
                    rot_consistency_weight = 1.0
                    loss = loss + rot_consistency_loss * rot_consistency_weight
                    logging_output["rot_consistency_loss"] = rot_consistency_loss.data
                else:
                    logging_output["rot_consistency_loss"] = torch.tensor(0.0).data
                
# noised_token_acc = (logits_encoder[noised_tokens].argmax(dim=-1) == token_label[noised_tokens]).long().sum()/noised_tokens.sum()                
            else:
                coord_target = sample[target_key]['coord_target']
                coord_target = torch.bmm(coord_target, rot_mat)
                
                coord_loss = F.smooth_l1_loss(
                    pred_coord[masked_tokens].view(-1, 3).float(),
                    coord_target[masked_tokens].view(-1, 3),
                    reduction="mean",
                    beta=1.0,
                )
                coord_loss_log_version = F.l1_loss(
                    pred_coord[masked_tokens].view(-1, 3).float(),
                    coord_target[masked_tokens].view(-1, 3),
                    reduction="mean",
                )
                loss = loss + coord_loss * self.args.masked_coord_loss * 10
                # restore the scale of loss for displaying
                logging_output["coord_loss"] = coord_loss_log_version.data
            
            
        else:
            coord_target = sample[target_key]['coord_target']
            coord_loss = F.smooth_l1_loss(
                encoder_coord[masked_tokens].view(-1, 3).float(),
                coord_target[masked_tokens].view(-1, 3),
                reduction="mean",
                beta=1.0,
            )
            coord_loss_log_version = F.l1_loss(
                encoder_coord[masked_tokens].view(-1, 3).float(),
                coord_target[masked_tokens].view(-1, 3),
                reduction="mean",
            )
            loss = loss + coord_loss * self.args.masked_coord_loss * 10
            # restore the scale of loss for displaying
            logging_output["coord_loss"] = coord_loss_log_version.data
        #loss = weight_re
        if encoder_distance is not None:
            dist_masked_tokens = masked_tokens
            masked_dist_loss, masked_dist_loss_log = self.cal_dist_loss_v2(
                sample, encoder_distance, dist_masked_tokens, target_key, normalize=True
            )
            loss = loss + masked_dist_loss * 10 * self.args.masked_dist_loss #+ gram_loss * 10
            logging_output["masked_dist_loss"] = masked_dist_loss_log.data
            #logging_output["gram_loss"] = gram_loss.data
        
            
        if self.args.encoder_x_norm_loss > 0 and encoder_x_norm is not None:
            loss = loss + self.args.encoder_x_norm_loss * encoder_x_norm
            logging_output["encoder_x_norm_loss"] = encoder_x_norm.data

        if mode == 'ae_only':
            #prop_pred = None
            if prop_pred is not None:
                #property_label = torch.tensor([compute_admet_properties(s)[1] for s in sample['target']['smi_name']])
                property_label = torch.tensor(np.array([self.property_table[s] for s in sample['target']['smi_name']])).float()
                #property_label = (property_label - prop_mean)/prop_std
                property_label = property_label.to(logits_encoder.device)
                prop_loss = F.mse_loss(prop_pred, property_label, reduction='mean')
                loss = loss + prop_loss #* self.args.property_pred_loss_weight
                logging_output['property_loss'] = prop_loss.data
        if z is not None:            
            
            """Default with Debug"""
            kl_div_element_wise = torch.distributions.kl.kl_divergence(q_z_given_x, p_z)
            
            kl_per_token = kl_div_element_wise.sum(dim=-1)
            
            #kl_divs = kl_div_element_wise[padding_mask]
            #kl_divs = kl_div_element_wise
            kl_loss = kl_per_token[padding_mask].mean()
            annealing_steps = 100000
            kl_weight = min(1.0, model.get_num_updates() / annealing_steps)
            #kl_weight = 1
            logging_output["KL_loss"] = kl_loss.data
            loss = loss + (kl_loss * kl_weight) * 0.01
            
        if (
            self.args.encoder_delta_pair_repr_norm_loss > 0
            and delta_encoder_pair_rep_norm is not None
        ):
            loss = (
                loss + self.args.encoder_delta_pair_repr_norm_loss * delta_encoder_pair_rep_norm
            )
            logging_output[
                "encoder_delta_pair_repr_norm_loss"
            ] = delta_encoder_pair_rep_norm.data

        if self.args.decoder_x_norm_loss > 0 and decoder_x_norm is not None:
            loss = loss + self.args.decoder_x_norm_loss * decoder_x_norm
            logging_output["decoder_x_norm_loss"] = decoder_x_norm.data

        if (
            self.args.decoder_delta_pair_repr_norm_loss > 0
            and delta_decoder_pair_rep_norm is not None
        ):
            loss = (
                loss + self.args.decoder_delta_pair_repr_norm_loss * delta_decoder_pair_rep_norm
            )
            logging_output[
                "decoder_delta_pair_repr_norm_loss"
            ] = delta_decoder_pair_rep_norm.data

        #logging_output["loss"] = loss.data
        
        """OT-CFM"""
        # logging_output["loss"] = flow_loss[masked_tokens].mean() 
        # loss = flow_loss[masked_tokens].mean() 
        
        if mode in ["ae_only", "dual"]:
            logging_output["loss"] = loss.data
        else:
            """FrameFlow"""
            logging_output["loss"] = flow_loss
            loss = flow_loss
            
                    
        return loss, 1, logging_output

    
    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        #metrics.log_scalar("seq_len", seq_len / bsz, 1, round=3)
        
        masked_loss = sum(log.get("token_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "token_loss", masked_loss / sample_size, sample_size, round=3
        )

        masked_acc = sum(
            log.get("masked_token_hit", 0) for log in logging_outputs
        ) / sum(log.get("masked_token_cnt", 0) for log in logging_outputs)
        metrics.log_scalar("masked_acc", masked_acc, sample_size, round=3)

        noised_token_loss = sum(log.get("noised_acc", 0) for log in logging_outputs)
        metrics.log_scalar(
            "noised_acc", noised_token_loss / sample_size, sample_size, round=3
        )
        
        bond_acc = sum(log.get("bond_acc", 0) for log in logging_outputs)
        metrics.log_scalar(
            "bond_acc", bond_acc / sample_size, sample_size, round=3
        )
        # Isbond_acc 
        
        Isbond_acc = sum(log.get("Isbond_acc", 0) for log in logging_outputs)
        metrics.log_scalar(
            "Isbond_acc", Isbond_acc / sample_size, sample_size, round=3
        )
        
        Notbond_acc = sum(log.get("Notbond_acc", 0) for log in logging_outputs)
        metrics.log_scalar(
            "Notbond_acc", Notbond_acc / sample_size, sample_size, round=3
        )
        
              
        coord_loss = sum(
            log.get("coord_loss", 0) for log in logging_outputs
        )
        if coord_loss > 0:
            metrics.log_scalar(
                "coord_loss",
                coord_loss / sample_size,
                sample_size,
                round=4,
            )
# 
        coord_refine_loss = sum(
            log.get("coord_refine_loss", 0) for log in logging_outputs
        )
        if coord_refine_loss > 0:
            metrics.log_scalar(
                "coord_refine_loss",
                coord_refine_loss / sample_size,
                sample_size,
                round=4,
            )

        rot_consistency_loss = sum(
            log.get("rot_consistency_loss", 0) for log in logging_outputs
        )
        if rot_consistency_loss > 0:
            metrics.log_scalar(
                "rot_consistency_loss",
                rot_consistency_loss / sample_size,
                sample_size,
                round=4,
            )
                                    
        bond_loss = sum(
            log.get("bond_loss", 0) for log in logging_outputs
        )
        if bond_loss > 0:
            metrics.log_scalar(
                "bond_loss",
                bond_loss / sample_size,
                sample_size,
                round=4,
            )
                        
        lm = sum(log.get("latent_mean", 0) for log in logging_outputs)
        ls = sum(log.get("latent_std", 0) for log in logging_outputs)
        metrics.log_scalar(
            "lat_mean", lm/sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "lat_std", ls/sample_size, sample_size, round=2
        )
                
        masked_dist_loss = sum(
            log.get("masked_dist_loss", 0) for log in logging_outputs
        )
        if masked_dist_loss > 0:
            metrics.log_scalar(
                "masked_dist_loss", masked_dist_loss / sample_size, sample_size, round=3
            )

        # gram_loss = sum(
        #     log.get("gram_loss", 0) for log in logging_outputs
        # )
        # if gram_loss > 0:
        #     metrics.log_scalar(
        #         "gram_loss", gram_loss / sample_size, sample_size, round=3
        #     )
        
        
        property_loss = sum(log.get("property_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "Property_loss", property_loss/sample_size, sample_size, round=5
        )        
            
        flow_loss = sum(log.get("flow_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "Flow_loss", flow_loss/sample_size, sample_size, round=5
        )

        f_1 = sum(log.get("f_loss[0,25)", 0) for log in logging_outputs)
        f_2 = sum(log.get("f_loss[25,50)", 0) for log in logging_outputs)
        f_3 = sum(log.get("f_loss[50,75)", 0) for log in logging_outputs)
        f_4 = sum(log.get("f_loss[75,100)", 0) for log in logging_outputs)

        metrics.log_scalar(
            "1st_Bin", f_1/sample_size, sample_size, round=2
        )
        metrics.log_scalar(
            "2nd_Bin", f_2/sample_size, sample_size, round=2
        )
        metrics.log_scalar(
            "3rd_Bin", f_3/sample_size, sample_size, round=2
        )
        metrics.log_scalar(
            "4th_Bin", f_4/sample_size, sample_size, round=2
        )        

                
        kl_loss = sum(log.get("KL_loss", 0) for log in logging_outputs)
        
        #if encoder_x_norm_loss > 0:
        metrics.log_scalar(
            "KL_loss", kl_loss / sample_size, sample_size, round=3
        )
        
        encoder_x_norm_loss = sum(log.get("encoder_x_norm_loss", 0) for log in logging_outputs)
        if encoder_x_norm_loss > 0:
            metrics.log_scalar(
                "encoder_x_norm_loss", encoder_x_norm_loss / sample_size, sample_size, round=3
            )

            
        encoder_delta_pair_repr_norm_loss = sum(
            log.get("encoder_delta_pair_repr_norm_loss", 0) for log in logging_outputs
        )
        if encoder_delta_pair_repr_norm_loss > 0:
            metrics.log_scalar(
                "encoder_delta_pair_repr_norm_loss",
                encoder_delta_pair_repr_norm_loss / sample_size,
                sample_size,
                round=3,
            )

        decoder_x_norm_loss = sum(log.get("decoder_x_norm_loss", 0) for log in logging_outputs)
        if decoder_x_norm_loss > 0:
            metrics.log_scalar(
                "decoder_x_norm_loss", decoder_x_norm_loss / sample_size, sample_size, round=3
            )

        decoder_delta_pair_repr_norm_loss = sum(
            log.get("decoder_delta_pair_repr_norm_loss", 0) for log in logging_outputs
        )
        if decoder_delta_pair_repr_norm_loss > 0:
            metrics.log_scalar(
                "decoder_delta_pair_repr_norm_loss",
                decoder_delta_pair_repr_norm_loss / sample_size,
                sample_size,
                round=3,
            )
        
        metrics.log_scalar("seq_len", seq_len / bsz, 1, round=3)
    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def cal_dist_loss(self, sample, dist, masked_tokens, target_key, normalize=False):
        dist_masked_tokens = masked_tokens
        masked_distance = dist[dist_masked_tokens, :]
        masked_distance_target = sample[target_key]["distance_target"][
            dist_masked_tokens
        ]
        non_pad_pos = masked_distance_target > 0
        if normalize:
            masked_distance_target = (
                masked_distance_target.float() - self.dist_mean
            ) / self.dist_std
        masked_dist_loss = F.smooth_l1_loss(
            masked_distance[non_pad_pos].view(-1).float(),
            masked_distance_target[non_pad_pos].view(-1),
            reduction="mean",
            beta=1.0,
        )
        return masked_dist_loss
    
    def cal_dist_loss_v2(self, sample, dist, masked_tokens, target_key, normalize=False):
        dist.diagonal(offset=0, dim1=-2, dim2=-1).fill_(0)
        dist_masked_tokens = masked_tokens.unsqueeze(-1) & masked_tokens.unsqueeze(-2)
        masked_distance = dist[dist_masked_tokens]
        masked_distance_target = sample['target']["distance_target"][
            dist_masked_tokens
        ]
        #non_pad_pos = masked_distance_target > 0
        # if normalize:
        #     masked_distance_target = (
        #         masked_distance_target.float() - self.dist_mean
        #     ) / self.dist_std
        masked_dist_loss = F.smooth_l1_loss(
            masked_distance.view(-1).float(),
            masked_distance_target.view(-1),
            reduction="mean",
            beta=1.0,
        )

        masked_dist_loss_log = F.l1_loss(
            masked_distance.view(-1).float(),
            masked_distance_target.view(-1),
            reduction="mean"
        )
                
        dist = dist * dist_masked_tokens
        #gram_loss = EDMLoss()(dist[:,1:,1:])
        return masked_dist_loss, masked_dist_loss_log
    
@register_loss("unimol_MAE_TypeC")
class UniMolOptimalLossTypeC(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        self.seed = task.seed
        self.dist_mean = 6.312581655060595
        self.dist_std = 3.3899264663911888
        self.idx = 0
        self.bond_extractor = CompleteBondExtractor(task)
        #self.bond_extractor = FixedCompleteBondExtractor(task)
        # with open('./unimol/example_data/total_dict.pickle', 'rb') as f:
        #     self.property_table = pickle.load(f)

        
    def forward(self, model, sample, reduce=True):
        self.idx += 1
        input_key = "net_input"
        target_key = "target"
        noised_tokens = sample[target_key]["tokens_target"].ne(self.padding_idx)

        
        #masked_tokens = None
        pair_padding_mask = sample[input_key]['src_distance'].ne(self.padding_idx)
        
        """At Generative Stage Only"""
        if self.task.args.task_sub == "generative":
            sample[input_key]['src_distance'] = sample[target_key]['distance_target']
            sample[input_key]['src_tokens'][sample[target_key]["tokens_target"] != 0] = sample[target_key]["tokens_target"][sample[target_key]["tokens_target"] != 0]
            sample[input_key]['src_coord'] = sample[target_key]['coord_target']
        
            
        sample[input_key]['src_tokens'][sample[input_key]['src_tokens'] == 1] = 0 
        sample[input_key]['src_tokens'][sample[input_key]['src_tokens'] == 2] = 0         
        sample = apply_random_rotation_to_target_coords(sample, target_key, apply_prob=0.7)
        
        
        padding_mask = sample[input_key]['src_tokens'].ne(self.padding_idx)
        
        masked_tokens = padding_mask

        bond_types = self.bond_extractor.extract_bond_types_from_sample(sample)
        sample_size = masked_tokens.long().sum()
        sample[input_key]['src_bond_type'] = bond_types
        
        
        # ae_only, flow_only, dual
        
        #mode = "dual"
        if self.task.args.training == "flow_only":
            mode = "flow_only"
        elif self.task.args.training == "ae_only":
            mode = "ae_only"
        else:
            mode = "dual"
        

        (
            logits_encoder,
            encoder_distance,
            encoder_coord,
            encoder_x_norm,
            decoder_x_norm,
            delta_encoder_pair_rep_norm,
            delta_decoder_pair_rep_norm,
            (z,q_z_given_x,p_z,latent_emb,std),
            flow_loss_dict,
            prop_pred
        ) = model(**sample[input_key], mode=mode, encoder_masked_tokens=None)
        
        # sample[input_key]['src_distance'] = sample[target_key]['distance_target']
        # sample[input_key]['src_tokens'][sample[target_key]["tokens_target"] != 0] = sample[target_key]["tokens_target"][sample[target_key]["tokens_target"] != 0]
        # sample[input_key]['src_coord'] = sample[target_key]['coord_target']
        token_label = sample[input_key]['src_tokens'].clone()
        token_label[sample[target_key]["tokens_target"] != 0] = sample[target_key]["tokens_target"][sample[target_key]["tokens_target"] != 0]
        weight_re = (1 - (self.idx % 73584)) / 73584
        weight_re = 0.
        #target = sample[target_key]["tokens_target"]
        if masked_tokens is not None:
            target = sample[input_key]['src_tokens'][masked_tokens]
            target = token_label[masked_tokens]
        token_loss = F.nll_loss(
            F.log_softmax(logits_encoder[padding_mask], dim=-1, dtype=torch.float32),
            target,
            ignore_index=self.padding_idx,
            reduction="mean",
        )
        
        noised_token_acc = (logits_encoder[noised_tokens].argmax(dim=-1) == token_label[noised_tokens]).long().sum()/noised_tokens.sum()
        masked_pred = logits_encoder[padding_mask].argmax(dim=-1)
        masked_hit = (masked_pred == target).long().sum()
        masked_cnt = sample_size
        loss = token_loss * 0.5 * self.args.masked_token_loss
        logging_output = {
            "sample_size": 1,
            "bsz": sample[target_key]["tokens_target"].size(0),
            "seq_len": sample[target_key]["tokens_target"].size(1)
            * sample[target_key]["tokens_target"].size(0),
            "token_loss": token_loss.data,
            "masked_token_hit": masked_hit.data,
            "masked_token_cnt": masked_cnt,
            #"latent_mean":latent_emb[masked_tokens].mean().data,
            #"latent_mean":torch.mean((torch.abs(latent_emb[masked_tokens]) < 0.2).float()).data,
            "latent_mean":torch.mean(torch.abs(latent_emb[masked_tokens])).data,
            #"latent_std":torch.mean()
            "noised_acc" : noised_token_acc.data,
            
            "latent_std":torch.mean((torch.abs(std[masked_tokens]) > 0.7).float()).data

            # "latent_mean":latent_emb.mean().data,
            # "latent_std":std.mean().data
        }
        
        """OT-CFM"""
        # #flow_loss = None
        # if flow_loss is not None:
        #     #flow_loss = flow_loss[masked_tokens].mean()
        #     loss = loss + flow_loss[masked_tokens].mean() 
        #     logging_output["flow_loss"] = flow_loss[masked_tokens].mean().data
        """FrameFlow"""
        if mode == "ae_only":
            flow_loss_dict = None
        
        if mode in ["flow_only", "dual"]:
            annealing_steps = 50000
            flow_weight = min(10.0, model.get_num_updates() / annealing_steps)
            #flow_weight = 10.
            flow_loss = flow_loss_dict['loss']
            loss = loss + flow_loss * flow_weight
            logging_output["flow_loss"] = flow_loss.data
            
            if "f_loss t=[0,25)" in flow_loss_dict.keys():
                logging_output["f_loss[0,25)"] = flow_loss_dict['f_loss t=[0,25)']
            else:
                logging_output["f_loss[0,25)"] = 0.
                
            if "f_loss t=[25,50)" in flow_loss_dict.keys():
                logging_output["f_loss[25,50)"] = flow_loss_dict['f_loss t=[25,50)']
            else:
                logging_output["f_loss[25,50)"] = 0.
                
            if "f_loss t=[50,75)" in flow_loss_dict.keys():
                logging_output["f_loss[50,75)"] = flow_loss_dict['f_loss t=[50,75)']
            else:
                logging_output["f_loss[50,75)"] = 0.
                
            if "f_loss t=[75,100)" in flow_loss_dict.keys():
                logging_output["f_loss[75,100)"] = flow_loss_dict['f_loss t=[75,100)']
            else:
                logging_output["f_loss[75,100)"] = 0.

        if len(encoder_coord)==2:

            rot_mat, pred_coord = encoder_coord
            if len(pred_coord) == 2:
                pred_coord_, pred_bond = pred_coord
                coord_target = sample[target_key]['coord_target']
                #coord_target = torch.bmm(coord_target, rot_mat)
                
                coord_loss = F.smooth_l1_loss(
                    pred_coord_[masked_tokens].view(-1, 3).float(),
                    coord_target[masked_tokens].view(-1, 3),
                    reduction="mean",
                    beta=1.0,
                )
                coord_loss_log_version = F.l1_loss(
                    pred_coord_[masked_tokens].view(-1, 3).float(),
                    coord_target[masked_tokens].view(-1, 3),
                    reduction="mean",
                )
                loss = loss + coord_loss * self.args.masked_coord_loss * 10
                # restore the scale of loss for displaying
                logging_output["coord_loss"] = coord_loss_log_version.data
                
                
                dist_masked_tokens = masked_tokens.unsqueeze(-1) & masked_tokens.unsqueeze(-2)
               
                bond_weight = torch.tensor([1,100,200,200,100], device=bond_types.device).float()
                bond_criterion = nn.CrossEntropyLoss(weight=bond_weight, reduction='mean')
                b_loss = bond_criterion(pred_bond[dist_masked_tokens], bond_types[dist_masked_tokens])
                
                loss = loss + b_loss
                logging_output["bond_loss"] = b_loss.data
                bond_acc = (pred_bond[dist_masked_tokens].argmax(-1) == bond_types[dist_masked_tokens]).long().sum()/dist_masked_tokens.sum()
                #bond_acc = (pred_bond[dist_masked_tokens].argmax(-1) == bond_types[dist_masked_tokens]).long().sum()/dist_masked_tokens.sum()
                is_bond = bond_types[dist_masked_tokens] != 0
                isbond_acc = (pred_bond[dist_masked_tokens].argmax(-1)[is_bond] == bond_types[dist_masked_tokens][is_bond]).long().sum()/is_bond.sum()
                
                isnot_bond = bond_types[dist_masked_tokens] == 0
                isnotbond_acc = (pred_bond[dist_masked_tokens].argmax(-1)[isnot_bond] == bond_types[dist_masked_tokens][isnot_bond]).long().sum()/isnot_bond.sum()
                
                logging_output['bond_acc'] = bond_acc.data
                logging_output['Isbond_acc'] = isbond_acc.data
                logging_output['Notbond_acc'] = isnotbond_acc.data

            elif len(pred_coord) == 3:
                pred_coord_refine, pred_bond, pred_coord_ = pred_coord
                coord_target = sample[target_key]['coord_target']
                coord_target = torch.bmm(coord_target, rot_mat)
                
                coord_loss = F.smooth_l1_loss(
                    pred_coord_[masked_tokens].view(-1, 3).float(),
                    coord_target[masked_tokens].view(-1, 3),
                    reduction="mean",
                    beta=1.0,
                )
                coord_loss_log_version = F.l1_loss(
                    pred_coord_[masked_tokens].view(-1, 3).float(),
                    coord_target[masked_tokens].view(-1, 3),
                    reduction="mean",
                )
                loss = loss + coord_loss * self.args.masked_coord_loss * 10
                # restore the scale of loss for displaying
                logging_output["coord_loss"] = coord_loss_log_version.data
                
                coord_refine_loss = F.smooth_l1_loss(
                    pred_coord_refine[masked_tokens].view(-1, 3).float(),
                    coord_target[masked_tokens].view(-1, 3),
                    reduction="mean",
                    beta=1.0,
                )
                coord_refine_loss_log_version = F.l1_loss(
                    pred_coord_refine[masked_tokens].view(-1, 3).float(),
                    coord_target[masked_tokens].view(-1, 3),
                    reduction="mean",
                )
                loss = loss + coord_loss * self.args.masked_coord_loss * 10 + coord_refine_loss * self.args.masked_coord_loss * 10
                
                # restore the scale of loss for displaying
                logging_output["coord_loss"] = coord_loss_log_version.data
                logging_output["coord_refine_loss"] = coord_refine_loss_log_version.data
                                
                dist_masked_tokens = masked_tokens.unsqueeze(-1) & masked_tokens.unsqueeze(-2)
               
                bond_weight = torch.tensor([1,100,200,200,100], device=bond_types.device).float()
                bond_criterion = nn.CrossEntropyLoss(weight=bond_weight, reduction='mean')
                b_loss = bond_criterion(pred_bond[dist_masked_tokens], bond_types[dist_masked_tokens])
                
                loss = loss + b_loss
                logging_output["bond_loss"] = b_loss.data
                bond_acc = (pred_bond[dist_masked_tokens].argmax(-1) == bond_types[dist_masked_tokens]).long().sum()/dist_masked_tokens.sum()
                #bond_acc = (pred_bond[dist_masked_tokens].argmax(-1) == bond_types[dist_masked_tokens]).long().sum()/dist_masked_tokens.sum()
                is_bond = bond_types[dist_masked_tokens] != 0
                isbond_acc = (pred_bond[dist_masked_tokens].argmax(-1)[is_bond] == bond_types[dist_masked_tokens][is_bond]).long().sum()/is_bond.sum()
                
                isnot_bond = bond_types[dist_masked_tokens] == 0
                isnotbond_acc = (pred_bond[dist_masked_tokens].argmax(-1)[isnot_bond] == bond_types[dist_masked_tokens][isnot_bond]).long().sum()/isnot_bond.sum()
                
                logging_output['bond_acc'] = bond_acc.data
                logging_output['Isbond_acc'] = isbond_acc.data
                logging_output['Notbond_acc'] = isnotbond_acc.data
                
                
# noised_token_acc = (logits_encoder[noised_tokens].argmax(dim=-1) == token_label[noised_tokens]).long().sum()/noised_tokens.sum()                
            else:
                coord_target = sample[target_key]['coord_target']
                coord_target = torch.bmm(coord_target, rot_mat)
                
                coord_loss = F.smooth_l1_loss(
                    pred_coord[masked_tokens].view(-1, 3).float(),
                    coord_target[masked_tokens].view(-1, 3),
                    reduction="mean",
                    beta=1.0,
                )
                coord_loss_log_version = F.l1_loss(
                    pred_coord[masked_tokens].view(-1, 3).float(),
                    coord_target[masked_tokens].view(-1, 3),
                    reduction="mean",
                )
                loss = loss + coord_loss * self.args.masked_coord_loss * 10
                # restore the scale of loss for displaying
                logging_output["coord_loss"] = coord_loss_log_version.data
            
            
        else:
            coord_target = sample[target_key]['coord_target']
            coord_loss = F.smooth_l1_loss(
                encoder_coord[masked_tokens].view(-1, 3).float(),
                coord_target[masked_tokens].view(-1, 3),
                reduction="mean",
                beta=1.0,
            )
            coord_loss_log_version = F.l1_loss(
                encoder_coord[masked_tokens].view(-1, 3).float(),
                coord_target[masked_tokens].view(-1, 3),
                reduction="mean",
            )
            loss = loss + coord_loss * self.args.masked_coord_loss * 10
            # restore the scale of loss for displaying
            logging_output["coord_loss"] = coord_loss_log_version.data
        #loss = weight_re
        if encoder_distance is not None:
            dist_masked_tokens = masked_tokens
            masked_dist_loss, masked_dist_loss_log = self.cal_dist_loss_v2(
                sample, encoder_distance, dist_masked_tokens, target_key, normalize=True
            )
            loss = loss + masked_dist_loss * 10 * self.args.masked_dist_loss #+ gram_loss * 10
            logging_output["masked_dist_loss"] = masked_dist_loss_log.data
            #logging_output["gram_loss"] = gram_loss.data
        
            
        if self.args.encoder_x_norm_loss > 0 and encoder_x_norm is not None:
            loss = loss + self.args.encoder_x_norm_loss * encoder_x_norm
            logging_output["encoder_x_norm_loss"] = encoder_x_norm.data

        if mode == 'ae_only':
            prop_pred = None
            if prop_pred is not None:
                #property_label = torch.tensor([compute_admet_properties(s)[1] for s in sample['target']['smi_name']])
                property_label = torch.tensor(np.array([self.property_table[s] for s in sample['target']['smi_name']])).float()
                #property_label = (property_label - prop_mean)/prop_std
                property_label = property_label.to(logits_encoder.device)
                prop_loss = F.mse_loss(prop_pred, property_label, reduction='mean')
                loss = loss + prop_loss #* self.args.property_pred_loss_weight
                logging_output['property_loss'] = prop_loss.data
        if z is not None:            
            
            """Default with Debug"""
            kl_div_element_wise = torch.distributions.kl.kl_divergence(q_z_given_x, p_z)
            
            kl_per_token = kl_div_element_wise.sum(dim=-1)
            
            #kl_divs = kl_div_element_wise[padding_mask]
            #kl_divs = kl_div_element_wise
            kl_loss = kl_per_token[padding_mask].mean()
            annealing_steps = 100000
            kl_weight = min(1.0, model.get_num_updates() / annealing_steps)
            #kl_weight = 1
            logging_output["KL_loss"] = kl_loss.data
            loss = loss + (kl_loss * kl_weight) * 0.01
            
        if (
            self.args.encoder_delta_pair_repr_norm_loss > 0
            and delta_encoder_pair_rep_norm is not None
        ):
            loss = (
                loss + self.args.encoder_delta_pair_repr_norm_loss * delta_encoder_pair_rep_norm
            )
            logging_output[
                "encoder_delta_pair_repr_norm_loss"
            ] = delta_encoder_pair_rep_norm.data

        if self.args.decoder_x_norm_loss > 0 and decoder_x_norm is not None:
            loss = loss + self.args.decoder_x_norm_loss * decoder_x_norm
            logging_output["decoder_x_norm_loss"] = decoder_x_norm.data

        if (
            self.args.decoder_delta_pair_repr_norm_loss > 0
            and delta_decoder_pair_rep_norm is not None
        ):
            loss = (
                loss + self.args.decoder_delta_pair_repr_norm_loss * delta_decoder_pair_rep_norm
            )
            logging_output[
                "decoder_delta_pair_repr_norm_loss"
            ] = delta_decoder_pair_rep_norm.data

        #logging_output["loss"] = loss.data
        
        """OT-CFM"""
        # logging_output["loss"] = flow_loss[masked_tokens].mean() 
        # loss = flow_loss[masked_tokens].mean() 
        
        if mode in ["ae_only", "dual"]:
            logging_output["loss"] = loss.data
        else:
            """FrameFlow"""
            logging_output["loss"] = flow_loss
            loss = flow_loss
            
                    
        return loss, 1, logging_output

    
    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        #metrics.log_scalar("seq_len", seq_len / bsz, 1, round=3)
        
        masked_loss = sum(log.get("token_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "token_loss", masked_loss / sample_size, sample_size, round=3
        )

        masked_acc = sum(
            log.get("masked_token_hit", 0) for log in logging_outputs
        ) / sum(log.get("masked_token_cnt", 0) for log in logging_outputs)
        metrics.log_scalar("masked_acc", masked_acc, sample_size, round=3)

        noised_token_loss = sum(log.get("noised_acc", 0) for log in logging_outputs)
        metrics.log_scalar(
            "noised_acc", noised_token_loss / sample_size, sample_size, round=3
        )
        
        bond_acc = sum(log.get("bond_acc", 0) for log in logging_outputs)
        metrics.log_scalar(
            "bond_acc", bond_acc / sample_size, sample_size, round=3
        )
        # Isbond_acc 
        
        Isbond_acc = sum(log.get("Isbond_acc", 0) for log in logging_outputs)
        metrics.log_scalar(
            "Isbond_acc", Isbond_acc / sample_size, sample_size, round=3
        )
        
        Notbond_acc = sum(log.get("Notbond_acc", 0) for log in logging_outputs)
        metrics.log_scalar(
            "Notbond_acc", Notbond_acc / sample_size, sample_size, round=3
        )
        
              
        coord_loss = sum(
            log.get("coord_loss", 0) for log in logging_outputs
        )
        if coord_loss > 0:
            metrics.log_scalar(
                "coord_loss",
                coord_loss / sample_size,
                sample_size,
                round=4,
            )

        coord_refine_loss = sum(
            log.get("coord_refine_loss", 0) for log in logging_outputs
        )
        if coord_refine_loss > 0:
            metrics.log_scalar(
                "coord_refine_loss",
                coord_refine_loss / sample_size,
                sample_size,
                round=4,
            )
                        
        bond_loss = sum(
            log.get("bond_loss", 0) for log in logging_outputs
        )
        if bond_loss > 0:
            metrics.log_scalar(
                "bond_loss",
                bond_loss / sample_size,
                sample_size,
                round=4,
            )
                        
        lm = sum(log.get("latent_mean", 0) for log in logging_outputs)
        ls = sum(log.get("latent_std", 0) for log in logging_outputs)
        metrics.log_scalar(
            "lat_mean", lm/sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "lat_std", ls/sample_size, sample_size, round=2
        )
                
        masked_dist_loss = sum(
            log.get("masked_dist_loss", 0) for log in logging_outputs
        )
        if masked_dist_loss > 0:
            metrics.log_scalar(
                "masked_dist_loss", masked_dist_loss / sample_size, sample_size, round=3
            )

        # gram_loss = sum(
        #     log.get("gram_loss", 0) for log in logging_outputs
        # )
        # if gram_loss > 0:
        #     metrics.log_scalar(
        #         "gram_loss", gram_loss / sample_size, sample_size, round=3
        #     )
        
        
        property_loss = sum(log.get("property_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "Property_loss", property_loss/sample_size, sample_size, round=5
        )        
            
        flow_loss = sum(log.get("flow_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "Flow_loss", flow_loss/sample_size, sample_size, round=5
        )

        f_1 = sum(log.get("f_loss[0,25)", 0) for log in logging_outputs)
        f_2 = sum(log.get("f_loss[25,50)", 0) for log in logging_outputs)
        f_3 = sum(log.get("f_loss[50,75)", 0) for log in logging_outputs)
        f_4 = sum(log.get("f_loss[75,100)", 0) for log in logging_outputs)

        metrics.log_scalar(
            "1st_Bin", f_1/sample_size, sample_size, round=2
        )
        metrics.log_scalar(
            "2nd_Bin", f_2/sample_size, sample_size, round=2
        )
        metrics.log_scalar(
            "3rd_Bin", f_3/sample_size, sample_size, round=2
        )
        metrics.log_scalar(
            "4th_Bin", f_4/sample_size, sample_size, round=2
        )        

                
        kl_loss = sum(log.get("KL_loss", 0) for log in logging_outputs)
        
        #if encoder_x_norm_loss > 0:
        metrics.log_scalar(
            "KL_loss", kl_loss / sample_size, sample_size, round=3
        )
        
        encoder_x_norm_loss = sum(log.get("encoder_x_norm_loss", 0) for log in logging_outputs)
        if encoder_x_norm_loss > 0:
            metrics.log_scalar(
                "encoder_x_norm_loss", encoder_x_norm_loss / sample_size, sample_size, round=3
            )

            
        encoder_delta_pair_repr_norm_loss = sum(
            log.get("encoder_delta_pair_repr_norm_loss", 0) for log in logging_outputs
        )
        if encoder_delta_pair_repr_norm_loss > 0:
            metrics.log_scalar(
                "encoder_delta_pair_repr_norm_loss",
                encoder_delta_pair_repr_norm_loss / sample_size,
                sample_size,
                round=3,
            )

        decoder_x_norm_loss = sum(log.get("decoder_x_norm_loss", 0) for log in logging_outputs)
        if decoder_x_norm_loss > 0:
            metrics.log_scalar(
                "decoder_x_norm_loss", decoder_x_norm_loss / sample_size, sample_size, round=3
            )

        decoder_delta_pair_repr_norm_loss = sum(
            log.get("decoder_delta_pair_repr_norm_loss", 0) for log in logging_outputs
        )
        if decoder_delta_pair_repr_norm_loss > 0:
            metrics.log_scalar(
                "decoder_delta_pair_repr_norm_loss",
                decoder_delta_pair_repr_norm_loss / sample_size,
                sample_size,
                round=3,
            )
        
        metrics.log_scalar("seq_len", seq_len / bsz, 1, round=3)
    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def cal_dist_loss(self, sample, dist, masked_tokens, target_key, normalize=False):
        dist_masked_tokens = masked_tokens
        masked_distance = dist[dist_masked_tokens, :]
        masked_distance_target = sample[target_key]["distance_target"][
            dist_masked_tokens
        ]
        non_pad_pos = masked_distance_target > 0
        if normalize:
            masked_distance_target = (
                masked_distance_target.float() - self.dist_mean
            ) / self.dist_std
        masked_dist_loss = F.smooth_l1_loss(
            masked_distance[non_pad_pos].view(-1).float(),
            masked_distance_target[non_pad_pos].view(-1),
            reduction="mean",
            beta=1.0,
        )
        return masked_dist_loss
    
    def cal_dist_loss_v2(self, sample, dist, masked_tokens, target_key, normalize=False):
        dist.diagonal(offset=0, dim1=-2, dim2=-1).fill_(0)
        dist_masked_tokens = masked_tokens.unsqueeze(-1) & masked_tokens.unsqueeze(-2)
        masked_distance = dist[dist_masked_tokens]
        masked_distance_target = sample['target']["distance_target"][
            dist_masked_tokens
        ]
        #non_pad_pos = masked_distance_target > 0
        # if normalize:
        #     masked_distance_target = (
        #         masked_distance_target.float() - self.dist_mean
        #     ) / self.dist_std
        masked_dist_loss = F.smooth_l1_loss(
            masked_distance.view(-1).float(),
            masked_distance_target.view(-1),
            reduction="mean",
            beta=1.0,
        )

        masked_dist_loss_log = F.l1_loss(
            masked_distance.view(-1).float(),
            masked_distance_target.view(-1),
            reduction="mean"
        )
                
        dist = dist * dist_masked_tokens
        #gram_loss = EDMLoss()(dist[:,1:,1:])
        return masked_dist_loss, masked_dist_loss_log
    
        
class EDMLoss(nn.Module):
    """
    Loss function that checks if a given squared distance matrix is a valid
    Euclidean Distance Matrix (EDM), and penalizes if not.

    This loss is computed based on eigenvalues of the Gram matrix,
    and only applies penalty when negative eigenvalues exist.
    """
    def __init__(self):
        super(EDMLoss, self).__init__()

    def forward(self, d_squared: torch.Tensor) -> torch.Tensor:
        """
        Args:
            d_squared (torch.Tensor): Model predicted squared distance matrix.
                                      Shape: [batch_size, num_atoms, num_atoms]

        Returns:
            torch.Tensor: Loss value for EDM constraint violation (scalar).
        """
        # Get batch_size and num_atoms from input tensor
        d_squared = d_squared**2
        batch_size, num_atoms, _ = d_squared.shape

        # 1. Create centering matrix J
        # J = I - (1/N) * 1 * 1^T
        # where N is num_atoms, I is identity matrix, 1 is all-ones vector
        identity = torch.eye(num_atoms, device=d_squared.device, dtype=d_squared.dtype)
        ones = torch.ones(num_atoms, num_atoms, device=d_squared.device, dtype=d_squared.dtype)
        j_matrix = identity - (1 / num_atoms) * ones

        # 2. Compute Gram matrix G using centering matrix J
        # G = -1/2 * J * D^2 * J
        # PyTorch's @ operator automatically handles batch matrix multiplication
        gram_matrix = -0.5 * (j_matrix @ d_squared @ j_matrix)
        gram_matrix = (gram_matrix + gram_matrix.transpose(-1, -2)) / 2

        # 3. Compute eigenvalues of Gram matrix
        # Since Gram matrix is symmetric, use torch.linalg.eigvalsh
        # to efficiently compute real eigenvalues only
        eigenvalues = torch.linalg.eigvalsh(gram_matrix)

        # 4. Penalize only negative eigenvalues
        # torch.relu(-eigenvalues) converts negative eigenvalues to positive,
        # and zeros out positive eigenvalues.
        # Divide sum of penalties by batch size to get mean loss.
        # Add small epsilon for numerical stability.
        #penalty = torch.relu(-eigenvalues).sum() / batch_size

        epsilon = 1e-8
        penalty = torch.relu(-eigenvalues - epsilon).sum(dim=1).mean()
        return penalty

import math

def generate_random_rotation_matrix(batch_size, device='cpu'):
    """
    Generate random 3D rotation matrices for each sample in the batch.
    
    Args:
        batch_size: Number of samples in the batch
        device: Device to put the tensor on
    
    Returns:
        rotation_matrices: [batch_size, 3, 3] random rotation matrices
    """
    # Generate random quaternions for efficient rotation matrix generation
    u = torch.rand(batch_size, device=device)
    v = torch.rand(batch_size, device=device) * 2 * math.pi
    w = torch.rand(batch_size, device=device) * 2 * math.pi

    # Convert to quaternion components
    q0 = torch.sqrt(1 - u) * torch.sin(v)
    q1 = torch.sqrt(1 - u) * torch.cos(v)
    q2 = torch.sqrt(u) * torch.sin(w)
    q3 = torch.sqrt(u) * torch.cos(w)

    # Convert quaternion to rotation matrix
    rotation_matrices = torch.zeros(batch_size, 3, 3, device=device)

    rotation_matrices[:, 0, 0] = 1 - 2 * (q2**2 + q3**2)
    rotation_matrices[:, 0, 1] = 2 * (q1*q2 - q0*q3)
    rotation_matrices[:, 0, 2] = 2 * (q1*q3 + q0*q2)

    rotation_matrices[:, 1, 0] = 2 * (q1*q2 + q0*q3)
    rotation_matrices[:, 1, 1] = 1 - 2 * (q1**2 + q3**2)
    rotation_matrices[:, 1, 2] = 2 * (q2*q3 - q0*q1)

    rotation_matrices[:, 2, 0] = 2 * (q1*q3 - q0*q2)
    rotation_matrices[:, 2, 1] = 2 * (q2*q3 + q0*q1)
    rotation_matrices[:, 2, 2] = 1 - 2 * (q1**2 + q2**2)

    return rotation_matrices
    
def apply_random_rotation_to_target_coords(sample, target_key, apply_prob=0.5):
    """
    Apply random rotation to target coordinates.

    Args:
        sample: Sample dictionary containing molecular data
        target_key: Key for target data (typically "target")
        apply_prob: Probability of applying rotation (default: 0.5)

    Returns:
        sample: Modified sample with rotated target coordinates
        rotation_matrices: Applied rotation matrices [B, 3, 3] or None if not applied
        applied: Boolean indicating whether rotation was applied
    """
    input_key = "net_input"
    coords = sample[target_key]['coord_target']  # [batch_size, max_atoms, 3]
    coords_src = sample[input_key]['src_coord']
    batch_size, max_atoms, _ = coords.shape

    # Only apply rotation with given probability
    if torch.rand(1).item() > apply_prob:
        return sample, None, False

    # Generate random rotation matrices for each sample in batch
    rotation_matrices = generate_random_rotation_matrix(batch_size, device=coords.device)
    # Apply rotation to coordinates
    # coords: [B, N, 3], rotation_matrices: [B, 3, 3]
    # We need to do batch matrix multiplication
    rotated_coords = torch.bmm(coords, rotation_matrices.transpose(-1, -2))  # [B, N, 3]
    rotated_coords_src = torch.bmm(coords_src, rotation_matrices.transpose(-1, -2))  # [B, N, 3]
    # Update the target coordinates in sample
    sample[target_key]['coord_target'] = rotated_coords
    sample[input_key]['src_coord'] = rotated_coords_src
    return sample, rotation_matrices, True

import sys
import torch
import numpy as np
from rdkit import Chem

class CompleteBondExtractor:
    """Complete bond type extractor"""

    def __init__(self, task=None):
        self.task = task
        self.dictionary = task.dictionary if task else None
        self.padding_idx = task.dictionary.pad() if task else 0

        # Default token mapping
        self.token_to_symbol = self._build_token_mapping()

    def _build_token_mapping(self):
        """Build token -> atom symbol mapping"""
        if self.dictionary:
            mapping = {}
            for i in range(len(self.dictionary)):
                symbol = self.dictionary[i]
                if symbol not in ['<pad>', '<unk>', '<s>', '</s>', '<mask>']:
                    mapping[i] = symbol
                else:
                    mapping[i] = 'PAD'
            return mapping
        else:
            return {0: 'PAD', 1: 'C', 2: 'O', 3: 'N', 4: 'S', 5: 'F', 6: 'Cl', 7: 'Br'}

    def get_smiles_from_sample(self, sample):
        """Extract SMILES from sample (supports multiple methods)"""

        # Method 1: Directly included in sample
        if 'net_input' in sample and 'src_smiles' in sample['net_input']:
            return sample['net_input']['src_smiles']

        # Method 2: Included in target
        if 'target' in sample and 'smiles_target' in sample['target']:
            return sample['target']['smiles_target']

        # Method 3: Included at top level
        if 'smiles_batch' in sample:
            return sample['smiles_batch']

        # Method 4: Get from cache (requires index info)
        # if hasattr(self, 'smiles_cache') and 'indices' in sample:
        #     return self.smiles_cache.get_smiles_batch(sample['indices'])
        return sample['target']['smi_name']
        # Default: dummy SMILES
        #batch_size = sample['net_input']['src_tokens'].shape[0]
        #return ["C"] * batch_size

    def smiles_to_bond_matrix(self, smiles, max_atoms):
        """SMILES -> bond type matrix"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros((max_atoms, max_atoms), dtype=np.int32)

            # Add hydrogens (adjust based on data)
            mol = Chem.AddHs(mol)
            bond_matrix = np.zeros((max_atoms, max_atoms), dtype=np.int32)

            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if i+1 >= max_atoms or j+1 >= max_atoms:
                    continue

                bond_type = bond.GetBondType()
                if bond_type == Chem.rdchem.BondType.SINGLE:
                    bond_value = 1
                elif bond_type == Chem.rdchem.BondType.DOUBLE:
                    bond_value = 2
                elif bond_type == Chem.rdchem.BondType.TRIPLE:
                    bond_value = 3
                elif bond_type == Chem.rdchem.BondType.AROMATIC:
                    bond_value = 4
                else:
                    bond_value = 1
                
                bond_matrix[i+1, j+1] = bond_value
                bond_matrix[j+1, i+1] = bond_value

            return bond_matrix

        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return np.zeros((max_atoms, max_atoms), dtype=np.int32)

    def extract_bond_types_from_sample(self, sample):
        """Extract bond types from sample (SMILES auto-detection)"""

        # Extract SMILES data
        smiles_batch = self.get_smiles_from_sample(sample)

        # Tensor info
        src_tokens = sample['net_input']['src_tokens']
        batch_size, max_atoms = src_tokens.shape

        # Check batch size
        if len(smiles_batch) != batch_size:
            print(f"Warning: SMILES batch size {len(smiles_batch)} != sample batch size {batch_size}")
            smiles_batch = smiles_batch[:batch_size] + ["C"] * max(0, batch_size - len(smiles_batch))

        bond_matrices = []

        for batch_idx in range(batch_size):
            smiles = smiles_batch[batch_idx] if batch_idx < len(smiles_batch) else "C"

            # Generate bond type matrix
            bond_matrix = self.smiles_to_bond_matrix(smiles, max_atoms)

            # Apply padding mask
            tokens = src_tokens[batch_idx]
            valid_mask = (tokens != self.padding_idx)
            bond_matrix[~valid_mask.cpu().numpy(), :] = 0
            bond_matrix[:, ~valid_mask.cpu().numpy()] = 0

            bond_matrices.append(bond_matrix)

        # Convert to tensor
        bond_types = torch.tensor(np.stack(bond_matrices), dtype=torch.long)
        if src_tokens.is_cuda:
            bond_types = bond_types.to(src_tokens.device)

        return bond_types


class FixedCompleteBondExtractor:
    """
    Fixed bond type extractor

    Key modifications:
    - Generate bond matrix based on LMDB atom order
    - Map RDKit mol atoms to LMDB order
    """

    def __init__(self, task=None):
        self.task = task
        self.dictionary = task.dictionary if task else None
        self.padding_idx = task.dictionary.pad() if task else 0

        # Default token mapping
        self.token_to_symbol = self._build_token_mapping()

    def _build_token_mapping(self):
        """Build token -> atom symbol mapping"""
        if self.dictionary:
            mapping = {}
            for i in range(len(self.dictionary)):
                symbol = self.dictionary[i]
                if symbol not in ['<pad>', '<unk>', '<s>', '</s>', '<mask>']:
                    mapping[i] = symbol
                else:
                    mapping[i] = 'PAD'
            return mapping
        else:
            return {0: 'PAD', 1: 'C', 2: 'O', 3: 'N', 4: 'S', 5: 'F', 6: 'Cl', 7: 'Br'}

    def get_smiles_from_sample(self, sample):
        """Extract SMILES from sample"""
        if 'net_input' in sample and 'src_smiles' in sample['net_input']:
            return sample['net_input']['src_smiles']

        if 'target' in sample and 'smiles_target' in sample['target']:
            return sample['target']['smiles_target']

        if 'smiles_batch' in sample:
            return sample['smiles_batch']

        return sample['target']['smi_name']

    def get_atoms_from_sample(self, sample, batch_idx):
        """
        Extract actual atom order from sample

        Args:
            sample: Batch sample
            batch_idx: Index within batch

        Returns:
            List[str]: Atom symbols in LMDB order
        """
        # Method 1: atoms info directly included in sample
        if 'target' in sample and 'atoms' in sample['target']:
            atoms_batch = sample['target']['atoms']
            if batch_idx < len(atoms_batch):
                return atoms_batch[batch_idx]

        # Method 2: atoms in net_input
        if 'net_input' in sample and 'atoms' in sample['net_input']:
            atoms_batch = sample['net_input']['atoms']
            if batch_idx < len(atoms_batch):
                return atoms_batch[batch_idx]

        # Method 3: Trace back from src_tokens
        # Requires token -> symbol conversion
        if 'net_input' in sample and 'src_tokens' in sample['net_input']:
            src_tokens = sample['net_input']['src_tokens']
            if batch_idx < src_tokens.shape[0]:
                tokens = src_tokens[batch_idx]
                atoms = []
                for token_id in tokens:
                    token_id = token_id.item() if torch.is_tensor(token_id) else token_id
                    if token_id == self.padding_idx:
                        break
                    symbol = self.token_to_symbol.get(token_id, 'C')
                    if symbol != 'PAD':
                        atoms.append(symbol)
                return atoms

        return None

    def create_atom_mapping(self, rdkit_mol, lmdb_atoms):
        """
        Map RDKit mol atoms to LMDB atoms order

        Args:
            rdkit_mol: RDKit Mol object (with H)
            lmdb_atoms: List of atom symbols from LMDB

        Returns:
            dict: {lmdb_idx: rdkit_idx} mapping
            None if mapping fails
        """
        rdkit_atoms = [atom.GetSymbol() for atom in rdkit_mol.GetAtoms()]

        # Cannot map if atom counts differ
        if len(rdkit_atoms) != len(lmdb_atoms):
            return None

        # Simple case: order is identical
        if rdkit_atoms == lmdb_atoms:
            return {i: i for i in range(len(lmdb_atoms))}

        # Complex case: find optimal mapping
        # First check if element counts match
        from collections import Counter
        rdkit_count = Counter(rdkit_atoms)
        lmdb_count = Counter(lmdb_atoms)

        if rdkit_count != lmdb_count:
            print(f"  Warning: Atom composition mismatch!")
            print(f"    RDKit: {rdkit_count}")
            print(f"    LMDB:  {lmdb_count}")
            return None

        # Greedy matching: map closest atoms based on coordinates
        # Only map atoms of the same element type
        mapping = {}
        used_rdkit_indices = set()

        for lmdb_idx, lmdb_symbol in enumerate(lmdb_atoms):
            # Find RDKit indices with same element
            candidates = [
                i for i, sym in enumerate(rdkit_atoms)
                if sym == lmdb_symbol and i not in used_rdkit_indices
            ]

            if not candidates:
                print(f"  Warning: No matching RDKit atom for LMDB atom {lmdb_idx} ({lmdb_symbol})")
                return None

            # Select first candidate (can be improved with coordinate-based selection)
            rdkit_idx = candidates[0]
            mapping[lmdb_idx] = rdkit_idx
            used_rdkit_indices.add(rdkit_idx)

        return mapping

    def smiles_to_bond_matrix_with_atom_order(self, smiles, lmdb_atoms, max_atoms):
        """
        Generate bond matrix using SMILES and LMDB atom order

        Args:
            smiles: SMILES string
            lmdb_atoms: List of atom symbols from LMDB (correct order)
            max_atoms: Maximum number of atoms (for padding)

        Returns:
            np.ndarray: Bond matrix [max_atoms, max_atoms]
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros((max_atoms, max_atoms), dtype=np.int32)

            # Add hydrogens
            mol = Chem.AddHs(mol)

            # Create mapping based on LMDB atom order
            atom_mapping = self.create_atom_mapping(mol, lmdb_atoms)

            if atom_mapping is None:
                print(f"  Warning: Failed to create atom mapping for SMILES: {smiles}")
                return np.zeros((max_atoms, max_atoms), dtype=np.int32)

            # Initialize bond matrix
            bond_matrix = np.zeros((max_atoms, max_atoms), dtype=np.int32)

            # Reverse mapping: rdkit_idx -> lmdb_idx
            reverse_mapping = {v: k for k, v in atom_mapping.items()}

            # Generate bond matrix based on LMDB indices for each bond
            for bond in mol.GetBonds():
                rdkit_i = bond.GetBeginAtomIdx()
                rdkit_j = bond.GetEndAtomIdx()

                # Convert RDKit index to LMDB index
                lmdb_i = reverse_mapping.get(rdkit_i)
                lmdb_j = reverse_mapping.get(rdkit_j)

                if lmdb_i is None or lmdb_j is None:
                    continue

                # Range check
                if lmdb_i + 1 >= max_atoms or lmdb_j + 1 >= max_atoms:
                    continue

                # Determine bond type
                bond_type = bond.GetBondType()
                if bond_type == Chem.rdchem.BondType.SINGLE:
                    bond_value = 1
                elif bond_type == Chem.rdchem.BondType.DOUBLE:
                    bond_value = 2
                elif bond_type == Chem.rdchem.BondType.TRIPLE:
                    bond_value = 3
                elif bond_type == Chem.rdchem.BondType.AROMATIC:
                    bond_value = 4
                else:
                    bond_value = 1

                # Add to bond matrix (1-indexed)
                bond_matrix[lmdb_i + 1, lmdb_j + 1] = bond_value
                bond_matrix[lmdb_j + 1, lmdb_i + 1] = bond_value

            return bond_matrix

        except Exception as e:
            print(f"  Error processing SMILES {smiles}: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((max_atoms, max_atoms), dtype=np.int32)

    def extract_bond_types_from_sample(self, sample):
        """
        Extract bond types from sample (using LMDB atom ordering)

        This method is called from UniMolOptimalLoss2.forward()
        """
        # Extract SMILES data
        smiles_batch = self.get_smiles_from_sample(sample)

        # Tensor info
        src_tokens = sample['net_input']['src_tokens']
        batch_size, max_atoms = src_tokens.shape

        # Check batch size
        if len(smiles_batch) != batch_size:
            print(f"Warning: SMILES batch size {len(smiles_batch)} != sample batch size {batch_size}")
            smiles_batch = smiles_batch[:batch_size] + ["C"] * max(0, batch_size - len(smiles_batch))

        bond_matrices = []

        for batch_idx in range(batch_size):
            smiles = smiles_batch[batch_idx] if batch_idx < len(smiles_batch) else "C"

            # Get LMDB atom order
            lmdb_atoms = self.get_atoms_from_sample(sample, batch_idx)

            if lmdb_atoms is None:
                print(f"  Warning: Could not extract atoms for batch {batch_idx}, using fallback")
                # Fallback: extract from tokens
                tokens = src_tokens[batch_idx]
                lmdb_atoms = []
                for token_id in tokens:
                    token_id = token_id.item() if torch.is_tensor(token_id) else token_id
                    if token_id == self.padding_idx:
                        break
                    symbol = self.token_to_symbol.get(token_id, 'C')
                    if symbol != 'PAD':
                        lmdb_atoms.append(symbol)

            # Generate bond matrix using LMDB atom order
            bond_matrix = self.smiles_to_bond_matrix_with_atom_order(
                smiles, lmdb_atoms, max_atoms
            )

            # Apply padding mask
            tokens = src_tokens[batch_idx]
            valid_mask = (tokens != self.padding_idx)
            bond_matrix[~valid_mask.cpu().numpy(), :] = 0
            bond_matrix[:, ~valid_mask.cpu().numpy()] = 0

            bond_matrices.append(bond_matrix)

        # Convert to tensor
        bond_types = torch.tensor(np.stack(bond_matrices), dtype=torch.long)
        if src_tokens.is_cuda:
            bond_types = bond_types.to(src_tokens.device)

        return bond_types
    
    
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

def compute_admet_properties(smiles: str) -> dict:
    """
    Compute various ADMET-related properties from given SMILES.

    :param smiles: SMILES string representing the molecule
    :return: Dictionary with property names as keys and computed values as values
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES input.")

    props = {
        "ExactMolWt": Descriptors.ExactMolWt(mol),                     # Exact molecular weight
        "MolLogP": Crippen.MolLogP(mol),                               # logP (lipophilicity)
        "MolMR": Crippen.MolMR(mol),                                   # Molar refractivity
        "NumHBD": Descriptors.NumHDonors(mol),                         # Number of H-bond donors
        "NumHBA": Descriptors.NumHAcceptors(mol),                      # Number of H-bond acceptors
        "TPSA": rdMolDescriptors.CalcTPSA(mol),                        # Topological polar surface area
        "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),              # Labute accessible surface area
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),  # Number of rotatable bonds
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),        # Fraction of sp3 carbons
        "NumHeavyAtoms": rdMolDescriptors.CalcNumHeavyAtoms(mol),      # Number of heavy atoms
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),    # Number of aromatic rings
        "NumAromaticHeterocycles": rdMolDescriptors.CalcNumAromaticHeterocycles(mol),  # Number of aromatic heterocycles
        "NumAromaticCarbocycles": rdMolDescriptors.CalcNumAromaticCarbocycles(mol),    # Number of aromatic carbocycles
        "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),    # Number of heteroatoms
        "QED": QED.qed(mol),                                           # QED (drug-likeness) score
    }

    # Apply PAINS filter: detect substructures causing non-specific binding or false positives
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    pains_catalog = FilterCatalog(params)
    if pains_catalog.HasMatch(mol)  == True:
        props["PAINS"] = 1
    else:
        props["PAINS"] = 0       # PAINS alert flag (True/False)
    property_list = list(props.values())
    return props, property_list


# input_key = "net_input"
# target_key = "target"
# self.padding_idx = 0

# # 1) Token-based non-padding mask
# nonpad_mask = sample[target_key]["tokens_target"].ne(self.padding_idx)

# pair_padding_mask = sample[input_key]['src_distance'].ne(self.padding_idx)
# sample[input_key]['src_tokens'][sample[target_key]["tokens_target"] != 0] = \
# sample[target_key]["tokens_target"][sample[target_key]["tokens_target"] != 0]
# sample[input_key]['src_coord'] = sample[target_key]['coord_target']
# sample[input_key]['src_distance'] = sample[target_key]['distance_target']
# sample[input_key]['src_tokens'][sample[input_key]['src_tokens'] == 1] = 0
# sample[input_key]['src_tokens'][sample[input_key]['src_tokens'] == 2] = 0

# # Build non-padding mask based on src_tokens
# nonpad_mask = sample[input_key]['src_tokens'].ne(self.padding_idx)  # True: atom, False: padding
# sample_size = nonpad_mask.long().sum()

# padding_mask = ~nonpad_mask  # True: padding

# bond_types = None
# sample[input_key]['src_bond_type'] = bond_types
# mode = 'dual'

# (
#     logits_encoder,
#     encoder_distance,
#     encoder_coord,
#     encoder_x_norm,
#     decoder_x_norm,
#     delta_encoder_pair_rep_norm,
#     delta_decoder_pair_rep_norm,
#     (z,q_z_given_x,p_z,latent_emb,std),
#     flow_loss_dict,
#     prop_pred
# ) = model(**sample[input_key], mode=mode, encoder_masked_tokens=None)



# if len(encoder_coord)==2:
#     rot_mat, pred_coord_ = encoder_coord
#     if len(pred_coord_) == 2:
#         pred_coord, pred_bond = pred_coord_
#         #coord_target = sample[target_key]['coord_target']
#         coord_target = torch.bmm(sample[input_key]['src_coord'], rot_mat)
# logits = logits_encoder[nonpad_mask]  # [num_valid, vocab]
# vocab_dict = self.create_vocab_dict()
# distance_mask = nonpad_mask.unsqueeze(-1) & nonpad_mask.unsqueeze(-2)  # [B, N, N]
# final_coords = pred_coord[~padding_mask]
# final_type = [vocab_dict[i] for i in logits.argmax(1).tolist()]
# iteration = 1553000
# final_atom_num = (~padding_mask==True).sum(1)
# mol_list = save_batch_to_sdf_reconstruct(final_type, final_atom_num, final_coords, iteration)


# # Save SDF with model-generated coordinates/types
# #final_coords = sample[input_key]['src_coord']
# final_coords = coord_target[~padding_mask]
# final_type = [vocab_dict[i] for i in sample[input_key]['src_tokens'][~padding_mask].tolist()]
# final_atom_num = (~padding_mask==True).sum(1)
# iteration = 1553

# mol_list = save_batch_to_sdf_reconstruct(final_type, final_atom_num, final_coords, iteration)
