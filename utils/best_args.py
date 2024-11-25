best_args = {
    'fl_digits': {

        'fedpad': {
            'local_lr': 0.01,
            'num_classes': 10,
            'local_batch_size': 64,
            'Note': '+ MSE',
            'mu': 0.01,
        },


    },
    'fl_officecaltech': {

        'fedpad': {
            'local_lr': 0.001,
            'num_classes': 10,
            'local_batch_size': 64,
            'mu': 0.01,
            'Note': '+ MSE',

        },
    },
    'fl_officehome': {

        'fedpad': {
            'local_lr': 0.001,
            'local_batch_size': 64,
            'mu': 0.01,
            'Note': '+ MSE'
        },

    },
    'fl_PACS': {

        'fedpad': {
            'local_lr': 0.001,
            'num_classes': 7,
            'local_batch_size': 64,
            'mu': 0.01,
            'Note': '+ MSE'
        }

    }
}
