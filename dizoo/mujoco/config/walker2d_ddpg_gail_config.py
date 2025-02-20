from easydict import EasyDict

walker2d_ddpg_gail_default_config = dict(
    exp_name='walker2d_ddpg_gail',
    env=dict(
        env_id='Walker2d-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    reward_model=dict(
        type='gail',
        input_size=23,
        hidden_size=256,
        batch_size=64,
        learning_rate=1e-3,
        update_per_collect=100,
        expert_data_path='walker2d_ddpg/expert_data.pkl',
        load_path='walker2d_ddpg_gail/reward_model/ckpt/ckpt_best.pth.tar',  # state_dict of the reward model
        expert_load_path='walker2d_ddpg/ckpt/ckpt_best.pth.tar',  # path to the expert state_dict
        collect_count=100000,
    ),
    policy=dict(
        load_path='walker2d_ddpg_gail/ckpt/ckpt_best.pth.tar',  # state_dict of the policy
        cuda=True,
        on_policy=False,
        random_collect_size=25000,
        model=dict(
            obs_shape=17,
            action_shape=6,
            twin_critic=False,
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
            action_space='regression',
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_actor=1e-3,
            learning_rate_critic=1e-3,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            actor_update_freq=1,
            noise=False,
        ),
        collect=dict(
            n_sample=64,
            unroll_len=1,
            noise_sigma=0.1,
        ),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    )
)
walker2d_ddpg_gail_default_config = EasyDict(walker2d_ddpg_gail_default_config)
main_config = walker2d_ddpg_gail_default_config

walker2d_ddpg_gail_default_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='ddpg',
        import_names=['ding.policy.ddpg'],
    ),
    replay_buffer=dict(type='naive', ),
)
walker2d_ddpg_gail_default_create_config = EasyDict(walker2d_ddpg_gail_default_create_config)
create_config = walker2d_ddpg_gail_default_create_config
