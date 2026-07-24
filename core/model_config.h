// core/model_config.h — Canonical C-only model configuration contract.
#ifndef ORION_MODEL_CONFIG_H
#define ORION_MODEL_CONFIG_H

typedef struct {
    int n_layer;
    int n_head;
    int d_model;
    int head_dim;
    int hidden_dim;
    int vocab;
    int max_seq;
} OrionModelConfig;

#endif // ORION_MODEL_CONFIG_H
