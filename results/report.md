# Contradiction detection on ContraDoc

Test split: 624 documents. Thresholds tuned for F1 on the dev split.
AUROC brackets are 95% bootstrap intervals (1000 resamples of test documents). To compare two systems use the gap column instead: it resamples the same documents for both, so it is much tighter. A gap whose interval excludes 0 is a real difference; with many systems in one table, expect a few such results by chance.

## Candidate pairs

Pairwise systems only see the top-K TF-IDF pairs per document, so this caps their recall.

| K | Gold pair in candidates | Evidence sentence in candidates |
|---|---|---|
| 20 | 59.5% of 321 | 75.1% of 449 |

## Pair level (gold pairs vs hardest pair in clean documents)

| System | AUROC | Gap to best | F1 | Acc | Order gap | n test |
|---|---|---|---|---|---|---|
| perplexity/pplx-decider-v1.2 | 0.892 [0.862, 0.917] | best | 0.766 | 0.787 | 0.048 | 530 |
| nli:deberta-v3-large-tasksource-nli | 0.870 [0.837, 0.897] | 0.022 [-0.005, 0.050] | 0.723 | 0.775 | - | 530 |
| liquid/d1-ultra | 0.867 [0.831, 0.900] | 0.025 [-0.002, 0.053] | 0.778 | 0.830 | 0.030 | 530 |
| subq-snap-preview | 0.866 [0.827, 0.897] | 0.026 [0.003, 0.051] | 0.766 | 0.808 | 0.028 | 530 |
| typesafe/jev-1.13 | 0.860 [0.825, 0.894] | 0.032 [0.000, 0.061] | 0.738 | 0.800 | 0.012 | 530 |
| typesafe/jev-1.14 | 0.857 [0.821, 0.892] | 0.034 [0.003, 0.065] | 0.746 | 0.774 | 0.015 | 530 |
| microsoft/microsoft-decision-1 | 0.856 [0.821, 0.887] | 0.036 [0.007, 0.064] | 0.721 | 0.736 | 0.000 | 530 |
| perplexity/pplx-decider-flash-v1.2 | 0.851 [0.814, 0.883] | 0.041 [0.015, 0.066] | 0.734 | 0.768 | 0.059 | 530 |
| cloudflare/clef | 0.851 [0.816, 0.884] | 0.041 [0.013, 0.067] | 0.709 | 0.713 | 0.036 | 530 |
| inception/mercury-decide | 0.850 [0.814, 0.884] | 0.042 [0.016, 0.068] | 0.715 | 0.743 | 0.027 | 530 |
| nli:roberta-large-mnli | 0.842 [0.806, 0.874] | 0.050 [0.019, 0.084] | 0.712 | 0.760 | - | 530 |
| nli:deberta-large-mnli | 0.833 [0.797, 0.867] | 0.059 [0.028, 0.091] | 0.685 | 0.757 | - | 530 |
| nli:bart-large-mnli | 0.833 [0.795, 0.867] | 0.059 [0.023, 0.095] | 0.692 | 0.753 | - | 530 |
| cloudflare/clef-omni | 0.828 [0.791, 0.863] | 0.064 [0.033, 0.094] | 0.693 | 0.723 | 0.018 | 530 |
| cloudflare/clef-flash | 0.827 [0.787, 0.865] | 0.065 [0.031, 0.098] | 0.695 | 0.691 | 0.029 | 530 |
| togethercomputer/tev1-4b-experimental | 0.825 [0.785, 0.862] | 0.067 [0.036, 0.096] | 0.679 | 0.651 | 0.025 | 530 |
| nace-ai/drex-v1.5 | 0.824 [0.787, 0.860] | 0.068 [0.039, 0.096] | 0.699 | 0.721 | 0.039 | 530 |
| respan/span-02 | 0.824 [0.787, 0.864] | 0.068 [0.034, 0.098] | 0.698 | 0.796 | 0.048 | 530 |
| nli:nli-deberta-v3-base | 0.819 [0.782, 0.854] | 0.073 [0.037, 0.112] | 0.696 | 0.753 | - | 530 |
| nace-ai/drex-v2.0 | 0.816 [0.777, 0.855] | 0.075 [0.044, 0.104] | 0.691 | 0.717 | 0.042 | 530 |
| nli:roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli | 0.808 [0.769, 0.847] | 0.084 [0.047, 0.120] | 0.688 | 0.764 | - | 530 |
| nli:nli-deberta-v3-large | 0.805 [0.766, 0.844] | 0.087 [0.050, 0.123] | 0.700 | 0.770 | - | 530 |
| liquid/d1 | 0.804 [0.765, 0.845] | 0.088 [0.056, 0.118] | 0.678 | 0.723 | 0.034 | 530 |
| nli:nli-distilroberta-base | 0.794 [0.751, 0.831] | 0.098 [0.057, 0.143] | 0.685 | 0.706 | - | 530 |
| nli:nli-deberta-v3-small | 0.793 [0.752, 0.831] | 0.099 [0.062, 0.138] | 0.691 | 0.730 | - | 530 |
| openai/gpt-6-luna-decisions | 0.791 [0.752, 0.832] | 0.101 [0.062, 0.137] | 0.661 | 0.772 | 0.065 | 530 |
| nli:deberta-base-long-nli | 0.790 [0.749, 0.827] | 0.102 [0.067, 0.140] | 0.639 | 0.704 | - | 530 |
| nli:nli-deberta-v3-xsmall | 0.789 [0.748, 0.827] | 0.103 [0.067, 0.143] | 0.670 | 0.719 | - | 530 |
| nli:nli-roberta-base | 0.785 [0.745, 0.827] | 0.107 [0.066, 0.149] | 0.654 | 0.721 | - | 530 |
| jaredpalmer/kev-4b | 0.780 [0.738, 0.821] | 0.112 [0.074, 0.147] | 0.667 | 0.725 | 0.030 | 530 |
| upstage/solar-decide-flash | 0.758 [0.718, 0.802] | 0.134 [0.095, 0.171] | 0.623 | 0.662 | 0.058 | 530 |
| upstage/solar-decide | 0.746 [0.701, 0.790] | 0.146 [0.105, 0.185] | 0.609 | 0.683 | 0.051 | 530 |

## Document level

| System | AUROC | Gap to best | F1 | Acc | P | R | Hit@1 | Hit@3 | Docs |
|---|---|---|---|---|---|---|---|---|---|
| always yes | - | - | 0.670 | 0.503 | 0.503 | 1.000 | - | - | - |
| liquid/d1-ultra [native:document] | 0.854 [0.822, 0.882] | best | 0.769 | 0.774 | 0.791 | 0.748 | 0.697 | 0.873 | 891 |
| perplexity/pplx-decider-v1.2 [native:max-sentence] | 0.851 [0.819, 0.883] | 0.003 [-0.020, 0.027] | 0.741 | 0.768 | 0.845 | 0.659 | 0.618 | 0.828 | 891 |
| perplexity/pplx-decider-v1.2 [native:document] | 0.846 [0.813, 0.877] | 0.008 [-0.012, 0.029] | 0.761 | 0.766 | 0.782 | 0.742 | 0.618 | 0.828 | 891 |
| liquid/d1-ultra [native:max-sentence] | 0.843 [0.812, 0.873] | 0.011 [-0.007, 0.026] | 0.753 | 0.750 | 0.748 | 0.758 | 0.697 | 0.873 | 891 |
| subq-snap-preview [native:max-sentence] | 0.838 [0.806, 0.868] | 0.016 [-0.010, 0.041] | 0.747 | 0.772 | 0.847 | 0.669 | 0.662 | 0.822 | 891 |
| subq-snap-preview [native:document] | 0.827 [0.795, 0.858] | 0.027 [0.005, 0.049] | 0.738 | 0.748 | 0.775 | 0.704 | 0.662 | 0.822 | 891 |
| inception/mercury-decide [native:document] | 0.797 [0.761, 0.832] | 0.057 [0.032, 0.082] | 0.703 | 0.628 | 0.588 | 0.873 | 0.510 | 0.790 | 891 |
| cloudflare/clef [native:max-sentence] | 0.792 [0.755, 0.825] | 0.062 [0.032, 0.090] | 0.729 | 0.726 | 0.726 | 0.732 | 0.599 | 0.780 | 891 |
| inception/mercury-decide [native:max-sentence] | 0.787 [0.751, 0.821] | 0.066 [0.037, 0.096] | 0.710 | 0.704 | 0.698 | 0.723 | 0.510 | 0.790 | 891 |
| typesafe/jev-1.13 [native:max-sentence] | 0.787 [0.751, 0.822] | 0.066 [0.037, 0.096] | 0.725 | 0.665 | 0.618 | 0.876 | 0.443 | 0.666 | 891 |
| typesafe/jev-1.14 [native:max-sentence] | 0.787 [0.753, 0.824] | 0.066 [0.036, 0.096] | 0.732 | 0.688 | 0.644 | 0.847 | 0.561 | 0.768 | 891 |
| typesafe/jev-1.13 [native:document] | 0.777 [0.741, 0.812] | 0.077 [0.049, 0.108] | 0.702 | 0.652 | 0.617 | 0.815 | 0.443 | 0.666 | 891 |
| typesafe/jev-1.14 [native:document] | 0.766 [0.728, 0.803] | 0.088 [0.060, 0.116] | 0.701 | 0.654 | 0.620 | 0.806 | 0.561 | 0.768 | 891 |
| respan/span-02 [native:max-sentence] | 0.766 [0.728, 0.804] | 0.088 [0.054, 0.120] | 0.701 | 0.704 | 0.711 | 0.691 | 0.462 | 0.666 | 891 |
| cloudflare/clef [native:document] | 0.764 [0.726, 0.800] | 0.090 [0.061, 0.121] | 0.685 | 0.671 | 0.662 | 0.710 | 0.599 | 0.780 | 891 |
| nace-ai/drex-v2.0 [native:max-sentence] | 0.756 [0.718, 0.794] | 0.097 [0.067, 0.129] | 0.670 | 0.503 | 0.503 | 1.000 | 0.401 | 0.611 | 891 |
| perplexity/pplx-decider-flash-v1.2 [native:document] | 0.741 [0.701, 0.779] | 0.113 [0.079, 0.146] | 0.696 | 0.635 | 0.599 | 0.831 | 0.363 | 0.596 | 891 |
| respan/span-02 [native:document] | 0.737 [0.700, 0.776] | 0.116 [0.084, 0.147] | 0.686 | 0.577 | 0.548 | 0.917 | 0.462 | 0.666 | 891 |
| perplexity/pplx-decider-flash-v1.2 [native:max-sentence] | 0.737 [0.694, 0.777] | 0.116 [0.082, 0.153] | 0.694 | 0.599 | 0.564 | 0.901 | 0.363 | 0.596 | 891 |
| microsoft/microsoft-decision-1 [native:document] | 0.734 [0.695, 0.772] | 0.120 [0.087, 0.152] | 0.675 | 0.519 | 0.511 | 0.994 | 0.404 | 0.653 | 891 |
| nace-ai/drex-v2.0 [native:document] | 0.725 [0.687, 0.765] | 0.128 [0.096, 0.162] | 0.689 | 0.554 | 0.531 | 0.981 | 0.401 | 0.611 | 891 |
| microsoft/microsoft-decision-1 [native:max-sentence] | 0.724 [0.682, 0.766] | 0.130 [0.092, 0.165] | 0.678 | 0.524 | 0.514 | 0.997 | 0.404 | 0.653 | 891 |
| liquid/d1 [native:max-sentence] | 0.723 [0.685, 0.764] | 0.131 [0.095, 0.167] | 0.676 | 0.540 | 0.524 | 0.952 | 0.452 | 0.634 | 891 |
| liquid/d1 [native:document] | 0.722 [0.682, 0.761] | 0.132 [0.097, 0.165] | 0.679 | 0.636 | 0.611 | 0.764 | 0.452 | 0.634 | 891 |
| cloudflare/clef-flash [native:max-sentence] | 0.713 [0.672, 0.754] | 0.141 [0.104, 0.177] | 0.665 | 0.614 | 0.590 | 0.761 | 0.411 | 0.583 | 891 |
| upstage/solar-decide [native:max-sentence] | 0.695 [0.654, 0.736] | 0.159 [0.115, 0.199] | 0.672 | 0.551 | 0.531 | 0.914 | 0.194 | 0.452 | 891 |
| nace-ai/drex-v1.5 [native:max-sentence] | 0.693 [0.652, 0.735] | 0.161 [0.123, 0.196] | 0.678 | 0.530 | 0.518 | 0.984 | 0.382 | 0.576 | 891 |
| togethercomputer/tev1-4b-experimental [native:max-sentence] | 0.689 [0.647, 0.728] | 0.164 [0.116, 0.210] | 0.684 | 0.583 | 0.553 | 0.895 | 0.306 | 0.478 | 891 |
| cloudflare/clef-flash [native:document] | 0.688 [0.648, 0.730] | 0.165 [0.127, 0.202] | 0.668 | 0.585 | 0.559 | 0.831 | 0.411 | 0.583 | 891 |
| cloudflare/clef-omni [native:document] | 0.685 [0.643, 0.726] | 0.169 [0.129, 0.207] | 0.670 | 0.532 | 0.519 | 0.943 | 0.140 | 0.287 | 891 |
| nace-ai/drex-v1.5 [native:document] | 0.673 [0.629, 0.716] | 0.181 [0.144, 0.218] | 0.656 | 0.538 | 0.525 | 0.876 | 0.382 | 0.576 | 891 |
| nli:deberta-base-long-nli [cand_k20] | 0.659 [0.618, 0.703] | 0.195 [0.149, 0.238] | 0.651 | 0.564 | 0.545 | 0.809 | 0.258 | 0.417 | 891 |
| nli:roberta-large-mnli [cand_k20] | 0.656 [0.614, 0.698] | 0.198 [0.152, 0.243] | 0.660 | 0.532 | 0.520 | 0.904 | 0.226 | 0.417 | 891 |
| togethercomputer/tev1-4b-experimental [native:document] | 0.653 [0.607, 0.692] | 0.201 [0.156, 0.248] | 0.699 | 0.588 | 0.553 | 0.949 | 0.306 | 0.478 | 891 |
| nli:deberta-v3-large-tasksource-nli [cand_k20] | 0.651 [0.606, 0.694] | 0.202 [0.160, 0.250] | 0.683 | 0.543 | 0.525 | 0.978 | 0.261 | 0.459 | 891 |
| jaredpalmer/kev-4b [native:max-sentence] | 0.650 [0.610, 0.693] | 0.204 [0.159, 0.245] | 0.676 | 0.535 | 0.521 | 0.962 | 0.306 | 0.446 | 891 |
| nli:deberta-large-mnli [cand_k20] | 0.649 [0.609, 0.694] | 0.205 [0.162, 0.248] | 0.681 | 0.538 | 0.522 | 0.978 | 0.229 | 0.411 | 891 |
| nli:roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli [cand_k20] | 0.645 [0.601, 0.686] | 0.208 [0.162, 0.254] | 0.676 | 0.530 | 0.518 | 0.971 | 0.226 | 0.424 | 891 |
| nli:nli-roberta-base [cand_k20] | 0.628 [0.585, 0.673] | 0.226 [0.177, 0.275] | 0.662 | 0.546 | 0.530 | 0.882 | 0.207 | 0.350 | 891 |
| cloudflare/clef-omni [native:max-sentence] | 0.627 [0.584, 0.669] | 0.226 [0.184, 0.268] | 0.674 | 0.526 | 0.515 | 0.975 | 0.140 | 0.287 | 891 |
| jaredpalmer/kev-4b [native:document] | 0.623 [0.578, 0.665] | 0.230 [0.189, 0.273] | 0.674 | 0.513 | 0.508 | 1.000 | 0.306 | 0.446 | 891 |
| nli:nli-deberta-v3-xsmall [cand_k20] | 0.618 [0.573, 0.664] | 0.236 [0.185, 0.287] | 0.671 | 0.508 | 0.506 | 0.997 | 0.210 | 0.376 | 891 |
| upstage/solar-decide [native:document] | 0.616 [0.575, 0.661] | 0.237 [0.190, 0.279] | 0.680 | 0.542 | 0.524 | 0.968 | 0.194 | 0.452 | 891 |
| nli:bart-large-mnli [cand_k20] | 0.611 [0.567, 0.652] | 0.242 [0.194, 0.294] | 0.665 | 0.556 | 0.536 | 0.876 | 0.213 | 0.385 | 891 |
| nli:nli-deberta-v3-base [cand_k20] | 0.602 [0.560, 0.647] | 0.252 [0.201, 0.301] | 0.670 | 0.521 | 0.513 | 0.965 | 0.213 | 0.366 | 891 |
| nli:nli-distilroberta-base [cand_k20] | 0.598 [0.551, 0.643] | 0.255 [0.205, 0.304] | 0.667 | 0.503 | 0.503 | 0.990 | 0.178 | 0.357 | 891 |
| nli:nli-deberta-v3-small [cand_k20] | 0.583 [0.541, 0.627] | 0.270 [0.214, 0.323] | 0.664 | 0.508 | 0.506 | 0.968 | 0.185 | 0.354 | 891 |
| nli:nli-deberta-v3-large [cand_k20] | 0.582 [0.538, 0.627] | 0.272 [0.224, 0.323] | 0.673 | 0.521 | 0.512 | 0.981 | 0.185 | 0.373 | 891 |
| upstage/solar-decide-flash [native:document] | 0.533 [0.491, 0.578] | 0.321 [0.273, 0.369] | 0.675 | 0.526 | 0.515 | 0.978 | 0.067 | 0.137 | 891 |
| upstage/solar-decide-flash [native:max-sentence] | 0.524 [0.480, 0.568] | 0.330 [0.278, 0.383] | 0.651 | 0.511 | 0.508 | 0.904 | 0.067 | 0.137 | 891 |

### Recall by contradiction type (test positives; multi-label)

| Type | cloudflare/clef [native:document] | cloudflare/clef [native:max-sentence] | cloudflare/clef-flash [native:document] | cloudflare/clef-flash [native:max-sentence] | cloudflare/clef-omni [native:document] | cloudflare/clef-omni [native:max-sentence] | inception/mercury-decide [native:document] | inception/mercury-decide [native:max-sentence] | jaredpalmer/kev-4b [native:document] | jaredpalmer/kev-4b [native:max-sentence] | liquid/d1 [native:document] | liquid/d1 [native:max-sentence] | liquid/d1-ultra [native:document] | liquid/d1-ultra [native:max-sentence] | microsoft/microsoft-decision-1 [native:document] | microsoft/microsoft-decision-1 [native:max-sentence] | nace-ai/drex-v1.5 [native:document] | nace-ai/drex-v1.5 [native:max-sentence] | nace-ai/drex-v2.0 [native:document] | nace-ai/drex-v2.0 [native:max-sentence] | nli:bart-large-mnli [cand_k20] | nli:deberta-base-long-nli [cand_k20] | nli:deberta-large-mnli [cand_k20] | nli:deberta-v3-large-tasksource-nli [cand_k20] | nli:nli-deberta-v3-base [cand_k20] | nli:nli-deberta-v3-large [cand_k20] | nli:nli-deberta-v3-small [cand_k20] | nli:nli-deberta-v3-xsmall [cand_k20] | nli:nli-distilroberta-base [cand_k20] | nli:nli-roberta-base [cand_k20] | nli:roberta-large-mnli [cand_k20] | nli:roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli [cand_k20] | perplexity/pplx-decider-flash-v1.2 [native:document] | perplexity/pplx-decider-flash-v1.2 [native:max-sentence] | perplexity/pplx-decider-v1.2 [native:document] | perplexity/pplx-decider-v1.2 [native:max-sentence] | respan/span-02 [native:document] | respan/span-02 [native:max-sentence] | subq-snap-preview [native:document] | subq-snap-preview [native:max-sentence] | togethercomputer/tev1-4b-experimental [native:document] | togethercomputer/tev1-4b-experimental [native:max-sentence] | typesafe/jev-1.13 [native:document] | typesafe/jev-1.13 [native:max-sentence] | typesafe/jev-1.14 [native:document] | typesafe/jev-1.14 [native:max-sentence] | upstage/solar-decide [native:document] | upstage/solar-decide [native:max-sentence] | upstage/solar-decide-flash [native:document] | upstage/solar-decide-flash [native:max-sentence] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Causal | 0.62 (n=29) | 0.62 (n=29) | 0.90 (n=29) | 0.72 (n=29) | 0.93 (n=29) | 0.93 (n=29) | 0.83 (n=29) | 0.48 (n=29) | 1.00 (n=29) | 0.97 (n=29) | 0.62 (n=29) | 0.93 (n=29) | 0.69 (n=29) | 0.76 (n=29) | 1.00 (n=29) | 1.00 (n=29) | 0.93 (n=29) | 1.00 (n=29) | 0.97 (n=29) | 1.00 (n=29) | 0.86 (n=29) | 0.69 (n=29) | 0.97 (n=29) | 0.97 (n=29) | 0.97 (n=29) | 1.00 (n=29) | 0.90 (n=29) | 1.00 (n=29) | 0.97 (n=29) | 0.83 (n=29) | 0.83 (n=29) | 0.97 (n=29) | 0.76 (n=29) | 0.83 (n=29) | 0.62 (n=29) | 0.52 (n=29) | 0.90 (n=29) | 0.62 (n=29) | 0.66 (n=29) | 0.45 (n=29) | 0.93 (n=29) | 0.86 (n=29) | 0.72 (n=29) | 0.90 (n=29) | 0.76 (n=29) | 0.79 (n=29) | 0.90 (n=29) | 0.86 (n=29) | 0.93 (n=29) | 0.83 (n=29) |
| Content | 0.71 (n=205) | 0.75 (n=205) | 0.83 (n=205) | 0.76 (n=205) | 0.92 (n=205) | 0.98 (n=205) | 0.86 (n=205) | 0.71 (n=205) | 1.00 (n=205) | 0.95 (n=205) | 0.77 (n=205) | 0.94 (n=205) | 0.76 (n=205) | 0.77 (n=205) | 0.99 (n=205) | 1.00 (n=205) | 0.88 (n=205) | 0.98 (n=205) | 0.98 (n=205) | 1.00 (n=205) | 0.87 (n=205) | 0.82 (n=205) | 0.97 (n=205) | 0.97 (n=205) | 0.95 (n=205) | 0.97 (n=205) | 0.96 (n=205) | 1.00 (n=205) | 0.99 (n=205) | 0.88 (n=205) | 0.89 (n=205) | 0.97 (n=205) | 0.83 (n=205) | 0.88 (n=205) | 0.74 (n=205) | 0.66 (n=205) | 0.90 (n=205) | 0.71 (n=205) | 0.69 (n=205) | 0.68 (n=205) | 0.94 (n=205) | 0.87 (n=205) | 0.82 (n=205) | 0.88 (n=205) | 0.80 (n=205) | 0.84 (n=205) | 0.96 (n=205) | 0.91 (n=205) | 0.97 (n=205) | 0.92 (n=205) |
| Emotion/Mood/Feeling | 0.62 (n=61) | 0.57 (n=61) | 0.87 (n=61) | 0.75 (n=61) | 0.93 (n=61) | 0.98 (n=61) | 0.89 (n=61) | 0.62 (n=61) | 1.00 (n=61) | 0.93 (n=61) | 0.64 (n=61) | 0.92 (n=61) | 0.67 (n=61) | 0.69 (n=61) | 0.98 (n=61) | 1.00 (n=61) | 0.79 (n=61) | 1.00 (n=61) | 1.00 (n=61) | 1.00 (n=61) | 0.85 (n=61) | 0.80 (n=61) | 0.98 (n=61) | 0.98 (n=61) | 0.95 (n=61) | 0.98 (n=61) | 0.98 (n=61) | 1.00 (n=61) | 1.00 (n=61) | 0.87 (n=61) | 0.92 (n=61) | 0.97 (n=61) | 0.84 (n=61) | 0.92 (n=61) | 0.74 (n=61) | 0.57 (n=61) | 0.95 (n=61) | 0.54 (n=61) | 0.62 (n=61) | 0.61 (n=61) | 0.97 (n=61) | 0.93 (n=61) | 0.75 (n=61) | 0.82 (n=61) | 0.69 (n=61) | 0.77 (n=61) | 0.97 (n=61) | 0.89 (n=61) | 0.95 (n=61) | 0.87 (n=61) |
| Factual | 0.79 (n=42) | 0.81 (n=42) | 0.79 (n=42) | 0.79 (n=42) | 1.00 (n=42) | 1.00 (n=42) | 0.90 (n=42) | 0.76 (n=42) | 1.00 (n=42) | 0.98 (n=42) | 0.88 (n=42) | 1.00 (n=42) | 0.79 (n=42) | 0.79 (n=42) | 1.00 (n=42) | 1.00 (n=42) | 0.90 (n=42) | 0.98 (n=42) | 0.95 (n=42) | 1.00 (n=42) | 0.88 (n=42) | 0.79 (n=42) | 0.93 (n=42) | 0.98 (n=42) | 0.95 (n=42) | 0.95 (n=42) | 0.95 (n=42) | 0.98 (n=42) | 1.00 (n=42) | 0.90 (n=42) | 0.86 (n=42) | 0.98 (n=42) | 0.86 (n=42) | 0.95 (n=42) | 0.76 (n=42) | 0.69 (n=42) | 0.93 (n=42) | 0.76 (n=42) | 0.81 (n=42) | 0.69 (n=42) | 0.90 (n=42) | 0.90 (n=42) | 0.86 (n=42) | 0.83 (n=42) | 0.83 (n=42) | 0.79 (n=42) | 0.98 (n=42) | 0.86 (n=42) | 1.00 (n=42) | 0.93 (n=42) |
| Negation | 0.83 (n=60) | 0.85 (n=60) | 0.88 (n=60) | 0.85 (n=60) | 0.97 (n=60) | 1.00 (n=60) | 0.90 (n=60) | 0.83 (n=60) | 1.00 (n=60) | 1.00 (n=60) | 0.85 (n=60) | 0.95 (n=60) | 0.85 (n=60) | 0.90 (n=60) | 1.00 (n=60) | 1.00 (n=60) | 0.97 (n=60) | 1.00 (n=60) | 0.97 (n=60) | 1.00 (n=60) | 0.95 (n=60) | 0.87 (n=60) | 0.98 (n=60) | 0.98 (n=60) | 1.00 (n=60) | 1.00 (n=60) | 1.00 (n=60) | 1.00 (n=60) | 1.00 (n=60) | 0.88 (n=60) | 0.95 (n=60) | 1.00 (n=60) | 0.88 (n=60) | 0.97 (n=60) | 0.85 (n=60) | 0.80 (n=60) | 0.92 (n=60) | 0.90 (n=60) | 0.80 (n=60) | 0.75 (n=60) | 0.98 (n=60) | 0.97 (n=60) | 0.92 (n=60) | 0.92 (n=60) | 0.88 (n=60) | 0.92 (n=60) | 0.97 (n=60) | 0.97 (n=60) | 0.98 (n=60) | 0.95 (n=60) |
| Numeric | 0.84 (n=49) | 0.82 (n=49) | 0.86 (n=49) | 0.82 (n=49) | 0.94 (n=49) | 1.00 (n=49) | 0.90 (n=49) | 0.82 (n=49) | 1.00 (n=49) | 0.98 (n=49) | 0.84 (n=49) | 1.00 (n=49) | 0.88 (n=49) | 0.88 (n=49) | 1.00 (n=49) | 1.00 (n=49) | 0.92 (n=49) | 0.98 (n=49) | 0.98 (n=49) | 1.00 (n=49) | 0.86 (n=49) | 0.90 (n=49) | 0.98 (n=49) | 1.00 (n=49) | 0.94 (n=49) | 0.98 (n=49) | 0.94 (n=49) | 1.00 (n=49) | 0.98 (n=49) | 0.92 (n=49) | 0.88 (n=49) | 0.96 (n=49) | 0.88 (n=49) | 0.96 (n=49) | 0.82 (n=49) | 0.80 (n=49) | 0.94 (n=49) | 0.82 (n=49) | 0.78 (n=49) | 0.80 (n=49) | 0.98 (n=49) | 0.92 (n=49) | 0.92 (n=49) | 0.94 (n=49) | 0.90 (n=49) | 0.94 (n=49) | 1.00 (n=49) | 0.96 (n=49) | 0.98 (n=49) | 0.88 (n=49) |
| Other | 1.00 (n=1) | 1.00 (n=1) | 0.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 0.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 0.00 (n=1) | 0.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 0.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) |
| Perspective/View/Opinion | 0.58 (n=66) | 0.61 (n=66) | 0.77 (n=66) | 0.67 (n=66) | 0.88 (n=66) | 0.94 (n=66) | 0.85 (n=66) | 0.68 (n=66) | 1.00 (n=66) | 0.94 (n=66) | 0.64 (n=66) | 0.94 (n=66) | 0.64 (n=66) | 0.67 (n=66) | 0.97 (n=66) | 0.98 (n=66) | 0.86 (n=66) | 0.97 (n=66) | 0.98 (n=66) | 1.00 (n=66) | 0.82 (n=66) | 0.68 (n=66) | 0.97 (n=66) | 0.91 (n=66) | 0.94 (n=66) | 0.97 (n=66) | 0.94 (n=66) | 1.00 (n=66) | 0.97 (n=66) | 0.82 (n=66) | 0.86 (n=66) | 0.92 (n=66) | 0.68 (n=66) | 0.74 (n=66) | 0.65 (n=66) | 0.50 (n=66) | 0.86 (n=66) | 0.62 (n=66) | 0.65 (n=66) | 0.59 (n=66) | 0.95 (n=66) | 0.89 (n=66) | 0.68 (n=66) | 0.86 (n=66) | 0.68 (n=66) | 0.86 (n=66) | 0.94 (n=66) | 0.92 (n=66) | 0.97 (n=66) | 0.92 (n=66) |
| Relation | 0.65 (n=17) | 0.65 (n=17) | 0.71 (n=17) | 0.53 (n=17) | 0.88 (n=17) | 0.94 (n=17) | 0.76 (n=17) | 0.59 (n=17) | 1.00 (n=17) | 1.00 (n=17) | 0.71 (n=17) | 0.88 (n=17) | 0.71 (n=17) | 0.65 (n=17) | 0.94 (n=17) | 1.00 (n=17) | 0.82 (n=17) | 1.00 (n=17) | 0.94 (n=17) | 1.00 (n=17) | 0.88 (n=17) | 0.88 (n=17) | 1.00 (n=17) | 0.94 (n=17) | 0.94 (n=17) | 0.94 (n=17) | 1.00 (n=17) | 1.00 (n=17) | 1.00 (n=17) | 0.88 (n=17) | 0.88 (n=17) | 0.94 (n=17) | 0.88 (n=17) | 0.71 (n=17) | 0.76 (n=17) | 0.71 (n=17) | 0.82 (n=17) | 0.47 (n=17) | 0.71 (n=17) | 0.82 (n=17) | 1.00 (n=17) | 0.71 (n=17) | 0.65 (n=17) | 0.94 (n=17) | 0.65 (n=17) | 0.71 (n=17) | 1.00 (n=17) | 1.00 (n=17) | 1.00 (n=17) | 0.88 (n=17) |

### Recall by scope

| Scope | cloudflare/clef [native:document] | cloudflare/clef [native:max-sentence] | cloudflare/clef-flash [native:document] | cloudflare/clef-flash [native:max-sentence] | cloudflare/clef-omni [native:document] | cloudflare/clef-omni [native:max-sentence] | inception/mercury-decide [native:document] | inception/mercury-decide [native:max-sentence] | jaredpalmer/kev-4b [native:document] | jaredpalmer/kev-4b [native:max-sentence] | liquid/d1 [native:document] | liquid/d1 [native:max-sentence] | liquid/d1-ultra [native:document] | liquid/d1-ultra [native:max-sentence] | microsoft/microsoft-decision-1 [native:document] | microsoft/microsoft-decision-1 [native:max-sentence] | nace-ai/drex-v1.5 [native:document] | nace-ai/drex-v1.5 [native:max-sentence] | nace-ai/drex-v2.0 [native:document] | nace-ai/drex-v2.0 [native:max-sentence] | nli:bart-large-mnli [cand_k20] | nli:deberta-base-long-nli [cand_k20] | nli:deberta-large-mnli [cand_k20] | nli:deberta-v3-large-tasksource-nli [cand_k20] | nli:nli-deberta-v3-base [cand_k20] | nli:nli-deberta-v3-large [cand_k20] | nli:nli-deberta-v3-small [cand_k20] | nli:nli-deberta-v3-xsmall [cand_k20] | nli:nli-distilroberta-base [cand_k20] | nli:nli-roberta-base [cand_k20] | nli:roberta-large-mnli [cand_k20] | nli:roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli [cand_k20] | perplexity/pplx-decider-flash-v1.2 [native:document] | perplexity/pplx-decider-flash-v1.2 [native:max-sentence] | perplexity/pplx-decider-v1.2 [native:document] | perplexity/pplx-decider-v1.2 [native:max-sentence] | respan/span-02 [native:document] | respan/span-02 [native:max-sentence] | subq-snap-preview [native:document] | subq-snap-preview [native:max-sentence] | togethercomputer/tev1-4b-experimental [native:document] | togethercomputer/tev1-4b-experimental [native:max-sentence] | typesafe/jev-1.13 [native:document] | typesafe/jev-1.13 [native:max-sentence] | typesafe/jev-1.14 [native:document] | typesafe/jev-1.14 [native:max-sentence] | upstage/solar-decide [native:document] | upstage/solar-decide [native:max-sentence] | upstage/solar-decide-flash [native:document] | upstage/solar-decide-flash [native:max-sentence] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global | 0.86 (n=105) | 0.86 (n=105) | 0.92 (n=105) | 0.87 (n=105) | 0.96 (n=105) | 1.00 (n=105) | 0.93 (n=105) | 0.85 (n=105) | 1.00 (n=105) | 0.99 (n=105) | 0.88 (n=105) | 0.96 (n=105) | 0.88 (n=105) | 0.87 (n=105) | 0.99 (n=105) | 1.00 (n=105) | 0.90 (n=105) | 1.00 (n=105) | 0.99 (n=105) | 1.00 (n=105) | 0.88 (n=105) | 0.88 (n=105) | 0.97 (n=105) | 0.99 (n=105) | 0.95 (n=105) | 0.97 (n=105) | 0.98 (n=105) | 0.99 (n=105) | 1.00 (n=105) | 0.89 (n=105) | 0.92 (n=105) | 0.98 (n=105) | 0.90 (n=105) | 0.93 (n=105) | 0.88 (n=105) | 0.83 (n=105) | 0.96 (n=105) | 0.81 (n=105) | 0.87 (n=105) | 0.86 (n=105) | 0.95 (n=105) | 0.91 (n=105) | 0.89 (n=105) | 0.91 (n=105) | 0.89 (n=105) | 0.90 (n=105) | 0.97 (n=105) | 0.94 (n=105) | 0.99 (n=105) | 0.90 (n=105) |
| global+intra | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) | 1.00 (n=1) |
| intra | 0.54 (n=54) | 0.59 (n=54) | 0.80 (n=54) | 0.76 (n=54) | 0.93 (n=54) | 0.96 (n=54) | 0.83 (n=54) | 0.54 (n=54) | 1.00 (n=54) | 0.98 (n=54) | 0.74 (n=54) | 0.96 (n=54) | 0.65 (n=54) | 0.59 (n=54) | 1.00 (n=54) | 1.00 (n=54) | 0.80 (n=54) | 0.98 (n=54) | 0.94 (n=54) | 1.00 (n=54) | 0.91 (n=54) | 0.72 (n=54) | 1.00 (n=54) | 1.00 (n=54) | 1.00 (n=54) | 1.00 (n=54) | 0.96 (n=54) | 1.00 (n=54) | 0.98 (n=54) | 0.87 (n=54) | 0.93 (n=54) | 1.00 (n=54) | 0.72 (n=54) | 0.89 (n=54) | 0.61 (n=54) | 0.46 (n=54) | 0.91 (n=54) | 0.52 (n=54) | 0.56 (n=54) | 0.39 (n=54) | 0.93 (n=54) | 0.87 (n=54) | 0.76 (n=54) | 0.83 (n=54) | 0.78 (n=54) | 0.78 (n=54) | 0.96 (n=54) | 0.83 (n=54) | 0.96 (n=54) | 0.91 (n=54) |
| local | 0.67 (n=154) | 0.69 (n=154) | 0.78 (n=154) | 0.69 (n=154) | 0.94 (n=154) | 0.96 (n=154) | 0.84 (n=154) | 0.70 (n=154) | 1.00 (n=154) | 0.94 (n=154) | 0.69 (n=154) | 0.94 (n=154) | 0.69 (n=154) | 0.74 (n=154) | 0.99 (n=154) | 0.99 (n=154) | 0.89 (n=154) | 0.97 (n=154) | 0.99 (n=154) | 1.00 (n=154) | 0.86 (n=154) | 0.79 (n=154) | 0.97 (n=154) | 0.96 (n=154) | 0.96 (n=154) | 0.98 (n=154) | 0.96 (n=154) | 1.00 (n=154) | 0.99 (n=154) | 0.88 (n=154) | 0.88 (n=154) | 0.95 (n=154) | 0.82 (n=154) | 0.88 (n=154) | 0.69 (n=154) | 0.61 (n=154) | 0.89 (n=154) | 0.67 (n=154) | 0.64 (n=154) | 0.64 (n=154) | 0.95 (n=154) | 0.89 (n=154) | 0.79 (n=154) | 0.86 (n=154) | 0.76 (n=154) | 0.84 (n=154) | 0.97 (n=154) | 0.92 (n=154) | 0.97 (n=154) | 0.90 (n=154) |

## Not reported (partial runs)

- nli:DeBERTa-v3-base-mnli-fever-anli [cand_k20]: 128/17815
- nli:DeBERTa-v3-large-mnli-fever-anli-ling-wanli [cand_k20]: 128/17815
- nli:mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 [cand_k20]: 128/17815
- nli:deberta-v2-xlarge-mnli [cand_k20]: 64/17815
- nli:deberta-v2-xxlarge-mnli [cand_k20]: 64/17815
- nli:deberta-xlarge-mnli [cand_k20]: 64/17815
- nli:albert-xlarge-vitaminc-mnli [cand_k20]: 64/17815
- openai/gpt-6-luna-decisions [native]: 890/891
- scaledown/classify-1 [pair_test]: 0/763
