import torch
import torch.nn as nn
import torch.nn.functional as F

from .knowformer_encoder import Encoder, get_param

pi = 3.14159265358979323846

class AngleScale:
    def __init__(self, embedding_range):
        self.embedding_range = embedding_range

    def __call__(self, axis_embedding, scale=None):
        if scale is None:
            scale = pi
        return axis_embedding / self.embedding_range * scale

def convert_to_axis(x):
    y = torch.tanh(x) * pi
    return y

def convert_to_arg(x):
    y = torch.tanh(2 * x) * pi / 2 + pi / 2
    return y

class ConeProjection(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers):
        super(ConeProjection, self).__init__()
        self.entity_dim = dim
        self.relation_dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer1 = nn.Linear(self.entity_dim + self.relation_dim, self.hidden_dim)
        self.layer0 = nn.Linear(self.hidden_dim, self.entity_dim + self.relation_dim)
        for nl in range(2, num_layers + 1):
            setattr(self, "layer{}".format(nl), nn.Linear(self.hidden_dim, self.hidden_dim))
        for nl in range(num_layers + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(nl)).weight)

    def forward(self, source_embedding_axis, source_embedding_arg, r_embedding_axis, r_embedding_arg):
        x = torch.cat([source_embedding_axis + r_embedding_axis, source_embedding_arg + r_embedding_arg], dim=-1)
        for nl in range(1, self.num_layers + 1):
            x = F.relu(getattr(self, "layer{}".format(nl))(x))
        x = self.layer0(x)

        axis, arg = torch.chunk(x, 2, dim=-1)
        axis_embeddings = convert_to_axis(axis)
        arg_embeddings = convert_to_arg(arg)
        return axis_embeddings, arg_embeddings

class ConeIntersection(nn.Module):
    def __init__(self, dim, drop):
        super(ConeIntersection, self).__init__()
        self.dim = dim
        self.layer_axis1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_arg1 = nn.Linear(self.dim * 2, self.dim)
        self.layer_axis2 = nn.Linear(self.dim, self.dim)
        self.layer_arg2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer_axis1.weight)
        nn.init.xavier_uniform_(self.layer_arg1.weight)
        nn.init.xavier_uniform_(self.layer_axis2.weight)
        nn.init.xavier_uniform_(self.layer_arg2.weight)

        self.drop = nn.Dropout(p=drop)

    def forward(self, axis_embeddings, arg_embeddings):
        logits = torch.cat([axis_embeddings - arg_embeddings, axis_embeddings + arg_embeddings], dim=-1)
        axis_layer1_act = F.relu(self.layer_axis1(logits))

        axis_attention = F.softmax(self.layer_axis2(axis_layer1_act), dim=0)

        x_embeddings = torch.cos(axis_embeddings)
        y_embeddings = torch.sin(axis_embeddings)
        x_embeddings = torch.sum(axis_attention * x_embeddings, dim=0)
        y_embeddings = torch.sum(axis_attention * y_embeddings, dim=0)

        x_embeddings[torch.abs(x_embeddings) < 1e-3] = 1e-3

        axis_embeddings = torch.atan(y_embeddings / x_embeddings)

        indicator_x = x_embeddings < 0
        indicator_y = y_embeddings < 0
        indicator_two = indicator_x & torch.logical_not(indicator_y)
        indicator_three = indicator_x & indicator_y

        axis_embeddings[indicator_two] = axis_embeddings[indicator_two] + pi
        axis_embeddings[indicator_three] = axis_embeddings[indicator_three] - pi

        # DeepSets
        arg_layer1_act = F.relu(self.layer_arg1(logits))
        arg_layer1_mean = torch.mean(arg_layer1_act, dim=0)
        gate = torch.sigmoid(self.layer_arg2(arg_layer1_mean))

        arg_embeddings = self.drop(arg_embeddings)
        arg_embeddings, _ = torch.min(arg_embeddings, dim=0)
        arg_embeddings = arg_embeddings * gate

        return axis_embeddings, arg_embeddings

class MergeEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super(MergeEmbedding, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(2*hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs):
        p, q = inputs
        lp = self.linear(p)
        lq = self.linear2(q)
        mid = nn.Sigmoid()(lq+lp)
        output = p * mid + lq * (1-mid)
        output = self.linear3(output)
        return output

class Knowformer(nn.Module):
    def __init__(self, config):
        super(Knowformer, self).__init__()
        self._emb_size = config['hidden_size']
        self._input_dropout_prob = config['input_dropout_prob']
        self._context_dropout_prob = config['context_dropout_prob']
        self._qual_dropout_prob = config['qual_dropout_prob']
        self._entity_dropout_prob = config['entity_dropout_prob']
        self.device = config['device']
        self.trm_moe = config['trm_moe']
        self.beta_weight = config['beta_weight']

        self._voc_size = config['vocab_size']
        self._entity_begin_idx = config['entity_begin_idx']
        self._entity_end_idx = config['entity_end_idx']
        self._relation_begin_idx = config['relation_begin_idx']
        self._relation_end_idx = config['relation_end_idx']
        self._special_begin_idx = config['special_begin_idx']
        self._special_end_idx = config['special_end_idx']

        self.ele_embedding = get_param((self._voc_size, self._emb_size))
        self.triple_encoder = Encoder(config, self.trm_moe)
        self.context_encoder = Encoder(config, self.trm_moe)
        self.input_dropout_layer = nn.Dropout(p=self._input_dropout_prob)
        self.context_dropout_layer = nn.Dropout(p=self._context_dropout_prob)
        self.qual_dropout_layer = nn.Dropout(p=self._qual_dropout_prob)
        self.entity_dropout_layer = torch.nn.Dropout(self._entity_dropout_prob)
        self.entity_or_relation_layernorm_layer = nn.LayerNorm(config['hidden_size'])

        self.angle_scale = AngleScale(config['initializer_range'])
        self.axis_scale = 1.0
        self.arg_scale = 1.0
        self.cone_proj = ConeProjection(self._emb_size, 800, 2)
        self.cone_intersection = ConeIntersection(self._emb_size, drop=0.1)
        self.axis_fc = torch.nn.Linear(5 * self._emb_size, self._emb_size)
        self.arg_fc = torch.nn.Linear(5 * self._emb_size, self._emb_size)
        self.rel_axis_fc = torch.nn.Linear(self._emb_size, self._emb_size)
        self.rel_arg_fc = torch.nn.Linear(self._emb_size, self._emb_size)

        self.merge_embedding = MergeEmbedding(config['hidden_size'])

    def __forward_triples(self, entity_embeddings, triple_ids, context_emb=None, qual_emb=None, encoder_mask=None, encoder_type="triple"):
        if entity_embeddings is None:
            entity_embeddings = torch.index_select(self.ele_embedding, 0, torch.tensor(range(self._entity_begin_idx, self._entity_end_idx)).to(self.device))
        entity_embeddings = self.entity_or_relation_layernorm_layer(entity_embeddings)
        relation_embeddings = torch.index_select(self.ele_embedding, 0, torch.tensor(range(self._relation_begin_idx, self._relation_end_idx)).to(self.device))
        relation_embeddings = self.entity_or_relation_layernorm_layer(relation_embeddings)
        relation_embeddings = self.entity_dropout_layer(relation_embeddings)
        special_embedding = torch.index_select(self.ele_embedding, 0, torch.tensor(range(self._special_begin_idx, self._special_end_idx)).to(self.device))
        ele_embedding = torch.cat((special_embedding, entity_embeddings, relation_embeddings), 0)

        batch_size, triple_len = triple_ids.shape[0], triple_ids.shape[1]
        emb_out = torch.index_select(ele_embedding, 0, triple_ids.view(-1)).view(batch_size, triple_len, -1)

        if context_emb is not None:
            context_emb = self.context_dropout_layer(context_emb)
            emb_out[:, 0, :] = (emb_out[:, 0, :] + context_emb) / 2
        if qual_emb is not None:
            qual_emb = self.qual_dropout_layer(qual_emb)
            emb_out[:, 1, :] = (emb_out[:, 1, :] + qual_emb) / 2

        emb_out = self.input_dropout_layer(emb_out)
        encoder = self.triple_encoder if encoder_type == "triple" else self.context_encoder
        emb_out = encoder(emb_out, mask=encoder_mask)
        return emb_out

    def __forward_qualifier_pairs_cone(self, entity_embeddings, tri_qual_input, tri_qual_ent_emb, tri_qual_rel_emb, rel_context_ids):
        if entity_embeddings is None:
            entity_embeddings = torch.index_select(self.ele_embedding, 0, torch.tensor(range(self._entity_begin_idx, self._entity_end_idx)).to(self.device))
        entity_embeddings = self.entity_or_relation_layernorm_layer(entity_embeddings)
        relation_embeddings = torch.index_select(self.ele_embedding, 0, torch.tensor(range(self._relation_begin_idx, self._relation_end_idx)).to(self.device))
        relation_embeddings = self.entity_or_relation_layernorm_layer(relation_embeddings)
        relation_embeddings = self.entity_dropout_layer(relation_embeddings)
        special_embedding = torch.index_select(self.ele_embedding, 0, torch.tensor(range(self._special_begin_idx, self._special_end_idx)).to(self.device))
        ele_embedding = torch.cat((special_embedding, entity_embeddings, relation_embeddings), 0)

        main_rel_axis_emb = self.rel_axis_fc(torch.index_select(ele_embedding, 0, tri_qual_input[:, 1]))
        main_rel_arg_emb = self.rel_arg_fc(torch.index_select(ele_embedding, 0, tri_qual_input[:, 1]))

        main_rel_axis_emb = self.angle_scale(main_rel_axis_emb, self.axis_scale)
        main_rel_arg_emb = self.angle_scale(main_rel_arg_emb, self.arg_scale)
        main_rel_axis_emb = convert_to_axis(main_rel_axis_emb)
        main_rel_arg_emb = convert_to_arg(main_rel_arg_emb)

        main_ent_axis_emb = self.angle_scale(tri_qual_ent_emb, self.axis_scale)
        main_ent_axis_emb = convert_to_axis(main_ent_axis_emb)
        main_ent_arg_emb = torch.zeros_like(main_ent_axis_emb).to(self.device)

        main_cone_axis, main_cone_arg = self.cone_proj(main_ent_axis_emb, main_ent_arg_emb, main_rel_axis_emb, main_rel_arg_emb)

        qualifier_rel_ids = rel_context_ids[:, :, 0]
        qualifier_ent_ids = rel_context_ids[:, :, 1]
        nei_qual_len, batch_size = rel_context_ids.shape[0], rel_context_ids.shape[1]
        qualifier_rel_emb = torch.index_select(ele_embedding, 0, qualifier_rel_ids.view(-1)).view(batch_size, nei_qual_len, -1)
        qualifier_ent_emb = torch.index_select(ele_embedding, 0, qualifier_ent_ids.view(-1)).view(batch_size, nei_qual_len, -1)

        query_cone_axis, query_cone_arg = self.shrinking_cone(main_cone_axis, main_cone_arg, tri_qual_rel_emb, main_rel_axis_emb, main_rel_arg_emb, qualifier_rel_emb, qualifier_ent_emb)

        return torch.cat((query_cone_axis, query_cone_arg), -1)

    def shrinking_cone(self, cones_axis, cones_arg, rel_emb, rel_axis_emb, rel_arg_emb, qual_rel_emb, qual_obj_emb):
        rel_embedded = rel_emb.unsqueeze(1).repeat(1, qual_obj_emb.shape[1], 1)
        rel_axis_embedded = rel_axis_emb.unsqueeze(1).repeat(1, qual_obj_emb.shape[1], 1)
        rel_arg_embedded = rel_arg_emb.unsqueeze(1).repeat(1, qual_obj_emb.shape[1], 1)

        rel_key_value_emb = torch.cat((rel_embedded, rel_axis_embedded, rel_arg_embedded, qual_rel_emb, qual_obj_emb), -1)

        repeat_cones_axis = cones_axis.unsqueeze(1).repeat(1, qual_obj_emb.shape[1], 1)
        repeat_cones_arg = cones_arg.unsqueeze(1).repeat(1, qual_obj_emb.shape[1], 1)

        fc_axis = self.axis_fc(rel_key_value_emb)
        fc_arg = self.arg_fc(rel_key_value_emb)

        shrinking_arg = F.sigmoid(fc_arg) * repeat_cones_arg
        offset = F.sigmoid(fc_axis) * (repeat_cones_arg - shrinking_arg)
        shrinking_axis = repeat_cones_axis - 0.5 * repeat_cones_arg + 0.5 * shrinking_arg + 0.5 * offset

        query_cones_axis = torch.cat([cones_axis.unsqueeze(1), shrinking_axis], dim=1)
        query_cones_arg = torch.cat([cones_arg.unsqueeze(1), shrinking_arg], dim=1)

        stack_query_cones_axis = query_cones_axis.permute(1, 0, 2)
        stack_query_cones_arg = query_cones_arg.permute(1, 0, 2)
        query_cones_axis, query_cones_arg = self.cone_intersection(stack_query_cones_axis, stack_query_cones_arg)

        return query_cones_axis, query_cones_arg

    def __process_mask_feat(self, mask_feat):
        return torch.matmul(mask_feat, self.ele_embedding.transpose(0, 1))

    def forward(self, tri_input, tri_qual_input, ent_context_input=None, ent_qual_context_input=None, rel_context_input=None, mask_pos=None, double_encoder=False):
        entity_embeddings = None
        tri_encoder_mask = (tri_input != 0).unsqueeze(1).unsqueeze(2)
        tri_seq_emb_out = self.__forward_triples(entity_embeddings, tri_input, context_emb=None, encoder_mask=tri_encoder_mask)
        tri_mask_emb = tri_seq_emb_out[:, 2, :]
        logits_from_triple = self.__process_mask_feat(tri_mask_emb)

        tri_qual_encoder_mask = (tri_qual_input != 0).unsqueeze(1).unsqueeze(2)
        tri_qual_seq_emb_out = self.__forward_triples(entity_embeddings, tri_qual_input, context_emb=None, encoder_mask=tri_qual_encoder_mask)
        batch_range = torch.tensor(torch.arange(0, end=tri_qual_seq_emb_out.shape[0]), dtype=torch.int64).view(-1).tolist()
        tri_qual_mask_emb = tri_qual_seq_emb_out[[batch_range], [mask_pos.view(-1).tolist()]].squeeze(0)
        logits_from_triple_qual = self.__process_mask_feat(tri_qual_mask_emb)

        if ent_context_input is not None:
            embeds_from_local_ent_neighbors = []
            logits_from_local_ent_neighbors = []
            for i in range(len(ent_context_input)):
                if double_encoder:
                    seq_emb_out = self.__forward_triples(entity_embeddings, ent_context_input[i], context_emb=None, qual_emb=None, encoder_type='context')
                else:
                    seq_emb_out = self.__forward_triples(entity_embeddings, ent_context_input[i], context_emb=None, qual_emb=None, encoder_type='triple')
                mask_emb = seq_emb_out[:, 0, :]
                logits = self.__process_mask_feat(mask_emb)
                embeds_from_local_ent_neighbors.append(mask_emb)
                logits_from_local_ent_neighbors.append(logits)

            context_embeds_ent_neighbors = torch.stack(embeds_from_local_ent_neighbors, dim=0)
            context_embeds_ent_neighbors = torch.mean(context_embeds_ent_neighbors, dim=0)
            logits_from_global_ent_neighbors = self.__process_mask_feat(context_embeds_ent_neighbors)

            context_local_logit = torch.stack(logits_from_local_ent_neighbors, dim=0)
            context_local_global_logit = torch.cat([context_local_logit, logits_from_global_ent_neighbors.unsqueeze(0)], dim=0)
            context_local_global_weight = torch.softmax(self.beta_weight * context_local_global_logit, dim=0)
            logits_from_local_global_ent_neighbors = (context_local_global_logit * context_local_global_weight.detach()).sum(0)

            context_local_logit = torch.stack(logits_from_local_ent_neighbors, dim=0)
            context_local_weight = torch.softmax(self.beta_weight * context_local_logit, dim=0)
            logits_from_local_ent_neighbors = (context_local_logit * context_local_weight.detach()).sum(0)

            tri_seq_emb_out = self.__forward_triples(entity_embeddings, tri_input, context_emb=context_embeds_ent_neighbors)
            tri_mask_emb = tri_seq_emb_out[:, 2, :]
            logits_from_triple_context = self.__process_mask_feat(tri_mask_emb)

        if ent_qual_context_input is not None:
            embeds_from_local_ent_qual_neighbors = []
            logits_from_local_ent_qual_neighbors = []
            for i in range(len(ent_qual_context_input)):
                if double_encoder:
                    seq_emb_out = self.__forward_triples(entity_embeddings, ent_qual_context_input[i], context_emb=None, qual_emb=None, encoder_type='context')
                else:
                    seq_emb_out = self.__forward_triples(entity_embeddings, ent_qual_context_input[i], context_emb=None, qual_emb=None, encoder_type='triple')
                mask_emb = seq_emb_out[:, 0, :]
                logits = self.__process_mask_feat(mask_emb)
                embeds_from_local_ent_qual_neighbors.append(mask_emb)
                logits_from_local_ent_qual_neighbors.append(logits)

            context_embeds_ent_qual_neighbors = torch.stack(embeds_from_local_ent_qual_neighbors, dim=0)
            context_embeds_ent_qual_neighbors = torch.mean(context_embeds_ent_qual_neighbors, dim=0)
            logits_from_global_ent_qual_neighbors = self.__process_mask_feat(context_embeds_ent_qual_neighbors)

            context_local_logit = torch.stack(logits_from_local_ent_qual_neighbors, dim=0)
            context_local_global_logit = torch.cat([context_local_logit, logits_from_global_ent_qual_neighbors.unsqueeze(0)], dim=0)
            context_local_global_weight = torch.softmax(self.beta_weight * context_local_global_logit, dim=0)
            logits_from_local_global_ent_qual_neighbors = (context_local_global_logit * context_local_global_weight.detach()).sum(0)

            context_local_logit = torch.stack(logits_from_local_ent_qual_neighbors, dim=0)
            context_local_weight = torch.softmax(self.beta_weight * context_local_logit, dim=0)
            logits_from_local_ent_qual_neighbors = (context_local_logit * context_local_weight.detach()).sum(0)

            tri_qual_seq_emb_out = self.__forward_triples(entity_embeddings, tri_qual_input, context_emb=context_embeds_ent_qual_neighbors)
            tri_qual_mask_emb = tri_qual_seq_emb_out[[batch_range], [mask_pos.view(-1).tolist()]].squeeze(0)

            if rel_context_input is not None:
                tri_qual_ent_emb = tri_qual_seq_emb_out[:, 0, :]
                tri_qual_rel_emb = tri_qual_seq_emb_out[:, 1, :]
                rel_context_ids = torch.stack(rel_context_input, dim=0)
                qualifier_pairs_neighbors_emb = self.__forward_qualifier_pairs_cone(entity_embeddings, tri_qual_input, tri_qual_ent_emb, tri_qual_rel_emb, rel_context_ids)
                tri_qual_mask_emb = self.merge_embedding([tri_qual_mask_emb, qualifier_pairs_neighbors_emb])

            logits_from_triple_qual_context = self.__process_mask_feat(tri_qual_mask_emb)

        return {
            'tri_without_neighbors': logits_from_triple,
            'tri_with_neighbors': logits_from_triple_context,
            'tri_local_neighbors': logits_from_local_ent_neighbors,
            'tri_global_neighbors': logits_from_global_ent_neighbors,
            'tri_local_global_neighbors': logits_from_local_global_ent_neighbors,

            'tri_qual_without_neighbors': logits_from_triple_qual,
            'tri_qual_with_neighbors': logits_from_triple_qual_context,
            'tri_qual_local_neighbors': logits_from_local_ent_qual_neighbors,
            'tri_qual_global_neighbors': logits_from_global_ent_qual_neighbors,
            'tri_qual_local_global_neighbors': logits_from_local_global_ent_qual_neighbors
        }