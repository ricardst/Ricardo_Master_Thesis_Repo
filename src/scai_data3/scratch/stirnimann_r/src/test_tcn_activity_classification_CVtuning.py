
            self.assertIsInstance(result, float)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)

    @patch('tcn_activity_classification_CVtuning.OPTIMIZE_SUBJECTS_PROBABILITY', 0.5)
    def test_subject_optimization_disabled(self):
        """Test objective when subject optimization is disabled"""
        self.mock_trial.suggest_float.side_effect = lambda name, low, high, **kwargs: {
            'optimize_subjects_prob': 0.8,  # Greater than OPTIMIZE_SUBJECTS_PROBABILITY
            'lr': 0.001,
            'weight_decay': 0.0001,
            'dropout_rate': 0.2,
            'scheduler_factor': 0.5,
            'scheduler_min_lr': 1e-6,
            'scheduler_eta_min': 1e-6
        }.get(name, (low + high) / 2)
        
        with patch('tcn_activity_classification_CVtuning.GroupKFold') as mock_gkf:
            mock_gkf_instance = Mock()
            mock_gkf_instance.split.return_value = [(np.array([0, 1]), np.array([2, 3]))]
            mock_gkf.return_value = mock_gkf_instance
            
            # Should use default subjects when optimization is disabled
            with patch.object(self.mock_trial, 'suggest_categorical') as mock_suggest_cat:
                mock_suggest_cat.side_effect = lambda name, choices: choices[0]
                
                result = objective(
                    self.mock_trial, 
                    self.all_available_data, 
                    self.fixed_test_subjects, 
                    self.device, 
        self.mock_trial.suggest_int.side_effect = lambda name, low, high: 2
        
        with patch('tcn_activity_classification_CVtuning.GroupKFold') as mock_gkf, \
             patch('tcn_activity_classification_CVtuning.StandardScaler'), \
             patch('tcn_activity_classification_CVtuning.LabelEncoder'), \
             patch('tcn_activity_classification_CVtuning.TCNModel'), \
             patch('tcn_activity_classification_CVtuning.DataLoader'), \
             patch('tcn_activity_classification_CVtuning.f1_score') as mock_f1:
            
            mock_gkf_instance = Mock()
            mock_gkf_instance.split.return_value = [(np.array([0, 1]), np.array([2, 3]))]
            mock_gkf.return_value = mock_gkf_instance
            
            mock_f1.return_value = 0.3
            
            self.mock_trial.report = Mock()
            self.mock_trial.should_prune.return_value = True  # Trigger pruning
            
            with self.assertRaises(optuna.TrialPruned):
                objective(
                    self.mock_trial, 
                    self.all_available_data, 
                    self.fixed_test_subjects, 
                    self.device, 
                    self.min_class_instances_config
                )

    def test_scheduler_configuration(self):
        """Test different scheduler configurations"""
        test_cases = [
            {'use_scheduler': True, 'scheduler_type': 'ReduceLROnPlateau'},
            {'use_scheduler': True, 'scheduler_type': 'CosineAnnealingLR'},
            {'use_scheduler': False, 'scheduler_type': 'ReduceLROnPlateau'}
        ]
        
        for case in test_cases:
            with self.subTest(case=case):
                self.mock_trial.suggest_categorical.side_effect = lambda name, choices: {
                    'use_scheduler': case['use_scheduler'],
                    'scheduler_type': case['scheduler_type'],
                    'tcn_kernel_size': 3,
                    'batch_size': 32
                }.get(name, choices[0])
                
                self.mock_trial.suggest_float.side_effect = lambda name, low, high, **kwargs: 0.5
                self.mock_trial.suggest_int.side_effect = lambda name, low, high: 2
                
                with patch('tcn_activity_classification_CVtuning.GroupKFold') as mock_gkf, \
                     patch('tcn_activity_classification_CVtuning.StandardScaler'), \
                     patch('tcn_activity_classification_CVtuning.LabelEncoder'), \
                     patch('tcn_activity_classification_CVtuning.TCNModel'), \
                     patch('tcn_activity_classification_CVtuning.DataLoader'), \
                     patch('tcn_activity_classification_CVtuning.f1_score') as mock_f1, \
            
            objective(
                self.mock_trial, 
                self.all_available_data, 
                self.fixed_test_subjects, 
                cuda_device, 
                self.min_class_instances_config
            )
            
            # Verify model was moved to correct device
            mock_model_instance.to.assert_called_with(cuda_device)

    def test_undersampling_logic(self):
        """Test the undersampling logic for imbalanced classes"""
        # Create imbalanced data
        imbalanced_data = self.all_available_data.copy()
        # Create class distribution: class1=100, class2=50, class3=200, class4=30
        y_imbalanced = (['class1'] * 100 + ['class2'] * 50 + 
                       ['class3'] * 200 + ['class4'] * 30 + 
                       ['class1'] * 620)  # Fill to 1000 total
        imbalanced_data['y_all'] = np.array(y_imbalanced)
        
        self.mock_trial.suggest_float.side_effect = lambda name, low, high, **kwargs: 0.5
        self.mock_trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
        self.mock_trial.suggest_int.side_effect = lambda name, low, high: 2
        
        with patch('tcn_activity_classification_CVtuning.GroupKFold') as mock_gkf, \
             patch('tcn_activity_classification_CVtuning.StandardScaler'), \
             patch('tcn_activity_classification_CVtuning.LabelEncoder'), \
             patch('tcn_activity_classification_CVtuning.TCNModel'), \
             patch('tcn_activity_classification_CVtuning.DataLoader'), \
             patch('tcn_activity_classification_CVtuning.f1_score') as mock_f1, \
             patch('numpy.random.choice') as mock_choice:
            
            mock_gkf_instance = Mock()
            mock_gkf_instance.split.return_value = [(np.array(list(range(500))), np.array(list(range(500, 600))))]
            mock_gkf.return_value = mock_gkf_instance
            
            # Mock random choice for undersampling
            mock_choice.side_effect = lambda arr, size, replace: arr[:size]
            