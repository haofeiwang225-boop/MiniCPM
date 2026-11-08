# CPU测试记录（第二阶段）

仅CPU合成/元张量；CUDA_VISIBLE_DEVICES为空。没有MiniCPM加载、真实梯度或正式实验。

暂存源码导出到干净临时目录后执行，排除未提交依赖。共29项通过。

```text
test_all_2000_schedule_boundaries (test_protocol.ProtocolTests) ... ok
test_cross_treatment_resume_rejected (test_protocol.ProtocolTests) ... ok
test_existing_output_rejected (test_protocol.ProtocolTests) ... ok
test_historical_lr_endpoints (test_protocol.ProtocolTests) ... ok
test_inventory_and_fail_closed (test_protocol.ProtocolTests) ... ok
test_84_real_shape_adapter_parameter_count_without_weights (test_runtime.RuntimeTests) ... ok
test_atomic_failure_no_published_checkpoint (test_runtime.RuntimeTests) ... ok
test_checkpoint_exact_next_step (test_runtime.RuntimeTests) ... ok
test_direct_official_equivalence_mixed_ranks_and_frozen_momentum (test_runtime.RuntimeTests) ... ok
test_dual_collector_cpu_once_per_sample (test_runtime.RuntimeTests) ... ok
test_metrics_distinct_fixed_dynamic_and_delta (test_runtime.RuntimeTests) ... ok
test_official_formats_and_paired_statistics (test_runtime.RuntimeTests) ... ok
test_pair_dropout_and_measurement_rng (test_runtime.RuntimeTests) ... ok
test_real_lr_scheduler_2000 (test_runtime.RuntimeTests) ... ok
test_real_official_schedule_2000 (test_runtime.RuntimeTests) ... ok
test_saved_order_arbitrary_record_count (test_runtime.RuntimeTests) ... ok
test_three_initial_adapters_and_base_output (test_runtime.RuntimeTests) ... ok

----------------------------------------------------------------------
Ran 17 tests in 1.909s

OK
test_actual_update_and_nonnegative_fit (test_gid_validation.GIDValidationTests) ... ok
test_entropy_uses_singular_values_not_energy (test_gid_validation.GIDValidationTests) ... ok
test_fixed_reference_cannot_be_overwritten (test_gid_validation.GIDValidationTests) ... ok
test_independent_random_budget_controls (test_gid_validation.GIDValidationTests) ... ok
test_initial_zero_b_active_oracle (test_gid_validation.GIDValidationTests) ... ok
test_nonfinite_rejected (test_gid_validation.GIDValidationTests) ... ok
test_positive_update_cannot_fit_descent (test_gid_validation.GIDValidationTests) ... ok
test_rank_deficient_factors (test_gid_validation.GIDValidationTests) ... ok
test_uniform_budget_impossible (test_gid_validation.GIDValidationTests) ... ok
test_zero_gradient_undefined (test_gid_validation.GIDValidationTests) ... ok
test_zero_spectrum (test_gid_validation.GIDValidationTests) ... ok

----------------------------------------------------------------------
Ran 11 tests in 0.053s

OK
test_eval_train_output_and_factor_gradients (test_historical_forward.HistoricalForwardTests) ... ok

----------------------------------------------------------------------
Ran 1 test in 0.048s

OK
historical_forward_error {"dtype": "torch.float32", "rank": 1, "training": false, "output": {"max_abs": 0.0, "max_relative": 0.0}, "A_grad": {"max_abs": 0.0, "max_relative": 0.0}, "B_grad": {"max_abs": 0.0, "max_relative": 0.0}}
historical_forward_error {"dtype": "torch.float32", "rank": 1, "training": true, "output": {"max_abs": 0.0, "max_relative": 0.0}, "A_grad": {"max_abs": 0.0, "max_relative": 0.0}, "B_grad": {"max_abs": 0.0, "max_relative": 0.0}}
historical_forward_error {"dtype": "torch.float32", "rank": 2, "training": false, "output": {"max_abs": 0.0, "max_relative": 0.0}, "A_grad": {"max_abs": 0.0, "max_relative": 0.0}, "B_grad": {"max_abs": 0.0, "max_relative": 0.0}}
historical_forward_error {"dtype": "torch.float32", "rank": 2, "training": true, "output": {"max_abs": 0.0, "max_relative": 0.0}, "A_grad": {"max_abs": 0.0, "max_relative": 0.0}, "B_grad": {"max_abs": 0.0, "max_relative": 0.0}}
historical_forward_error {"dtype": "torch.float32", "rank": 3, "training": false, "output": {"max_abs": 0.0, "max_relative": 0.0}, "A_grad": {"max_abs": 0.0, "max_relative": 0.0}, "B_grad": {"max_abs": 0.0, "max_relative": 0.0}}
historical_forward_error {"dtype": "torch.float32", "rank": 3, "training": true, "output": {"max_abs": 0.0, "max_relative": 0.0}, "A_grad": {"max_abs": 0.0, "max_relative": 0.0}, "B_grad": {"max_abs": 0.0, "max_relative": 0.0}}
historical_forward_error {"dtype": "torch.float32", "rank": 4, "training": false, "output": {"max_abs": 0.0, "max_relative": 0.0}, "A_grad": {"max_abs": 0.0, "max_relative": 0.0}, "B_grad": {"max_abs": 0.0, "max_relative": 0.0}}
historical_forward_error {"dtype": "torch.float32", "rank": 4, "training": true, "output": {"max_abs": 0.0, "max_relative": 0.0}, "A_grad": {"max_abs": 0.0, "max_relative": 0.0}, "B_grad": {"max_abs": 0.0, "max_relative": 0.0}}
historical_forward_error {"dtype": "torch.bfloat16", "rank": 1, "training": false, "output": {"max_abs": 0.0, "max_relative": 0.0}, "A_grad": {"max_abs": 0.0, "max_relative": 0.0}, "B_grad": {"max_abs": 0.0, "max_relative": 0.0}}
historical_forward_error {"dtype": "torch.bfloat16", "rank": 1, "training": true, "output": {"max_abs": 0.0, "max_relative": 0.0}, "A_grad": {"max_abs": 0.0, "max_relative": 0.0}, "B_grad": {"max_abs": 0.0, "max_relative": 0.0}}
historical_forward_error {"dtype": "torch.bfloat16", "rank": 2, "training": false, "output": {"max_abs": 0.0, "max_relative": 0.0}, "A_grad": {"max_abs": 0.0, "max_relative": 0.0}, "B_grad": {"max_abs": 0.0, "max_relative": 0.0}}
historical_forward_error {"dtype": "torch.bfloat16", "rank": 2, "training": true, "output": {"max_abs": 0.0, "max_relative": 0.0}, "A_grad": {"max_abs": 0.0, "max_relative": 0.0}, "B_grad": {"max_abs": 0.0, "max_relative": 0.0}}
historical_forward_error {"dtype": "torch.bfloat16", "rank": 3, "training": false, "output": {"max_abs": 0.0, "max_relative": 0.0}, "A_grad": {"max_abs": 0.0, "max_relative": 0.0}, "B_grad": {"max_abs": 0.0, "max_relative": 0.0}}
historical_forward_error {"dtype": "torch.bfloat16", "rank": 3, "training": true, "output": {"max_abs": 0.0, "max_relative": 0.0}, "A_grad": {"max_abs": 0.0, "max_relative": 0.0}, "B_grad": {"max_abs": 0.0, "max_relative": 0.0}}
historical_forward_error {"dtype": "torch.bfloat16", "rank": 4, "training": false, "output": {"max_abs": 0.0, "max_relative": 0.0}, "A_grad": {"max_abs": 0.0, "max_relative": 0.0}, "B_grad": {"max_abs": 0.0, "max_relative": 0.0}}
historical_forward_error {"dtype": "torch.bfloat16", "rank": 4, "training": true, "output": {"max_abs": 0.0, "max_relative": 0.0}, "A_grad": {"max_abs": 0.0, "max_relative": 0.0}, "B_grad": {"max_abs": 0.0, "max_relative": 0.0}}
```

只读真实JSON核验：84模块，实际480000参数，预算480256；train2000条，顺序2000步/2000唯一索引。
顺序清单canonical SHA256：59aebf0236f5a6bbc6fc7a0de48891f76e319e9f8e56cde7b260ad4620ace0bb。

PyTorch 2.5.1+cu121；AdamW defaults betas=(0.9,0.999), eps=1e-8。
命令见README.md。AST静态解析和bash -n通过；git diff --check仅官方原文件尾随空白，不为消除告警修改vendor。
