input_lvs = [
    {
        'X': (0, 26, 1),
        'name': 'Java skills',
        'terms': {
            'unskilled': {'umf': ('trapmf', 0, 0, 0.55, 4.61), 'lmf': ('trapmf', 0, 0, 0.09, 1.15, 1)},
            'more or less unskilled': {'umf': ('trapmf', 0.42, 2.25, 4.00, 5.41), 'lmf': ('trapmf', 2.79, 3.21, 3.21, 0.34, 3.71)},
            'more or less skilled': {'umf': ('trapmf', 3.38, 5.50, 7.25, 9.02), 'lmf': ('trapmf', 5.79, 6.28, 6.28, 0.33, 6.67)},
            'highly skilled': {'umf': ('trapmf', 7.37, 9.36, 10, 10), 'lmf': ('trapmf', 8.68, 9.91, 10, 10, 1)},
        }
    },

    {
        'X': (0, 61, 1),
        'name': 'English level',
        'terms': {
            'unskilled': {'umf': ('trapmf', 0, 0, 0.55, 4.61), 'lmf': ('trapmf', 0, 0, 0.09, 1.15, 1)},
            'more or less unskilled': {'umf': ('trapmf', 0.42, 2.25, 4.00, 5.41), 'lmf': ('trapmf', 2.79, 3.21, 3.21, 0.34, 3.71)},
            'more or less skilled': {'umf': ('trapmf', 3.38, 5.50, 7.25, 9.02), 'lmf': ('trapmf', 5.79, 6.28, 6.28, 0.33, 6.67)},
            'highly skilled': {'umf': ('trapmf', 7.37, 9.36, 10, 10), 'lmf': ('trapmf', 8.68, 9.91, 10, 10, 1)},
        }
    },

    {
        'X': (0, 51, 1),
        'name': 'Soft skills',
        'terms': {
            'unskilled': {'umf': ('trapmf', 0, 0, 0.55, 4.61), 'lmf': ('trapmf', 0, 0, 0.09, 1.15, 1)},
            'more or less unskilled': {'umf': ('trapmf', 0.42, 2.25, 4.00, 5.41), 'lmf': ('trapmf', 2.79, 3.21, 3.21, 0.34, 3.71)},
            'more or less skilled': {'umf': ('trapmf', 3.38, 5.50, 7.25, 9.02), 'lmf': ('trapmf', 5.79, 6.28, 6.28, 0.33, 6.67)},
            'highly skilled': {'umf': ('trapmf', 7.37, 9.36, 10, 10), 'lmf': ('trapmf', 8.68, 9.91, 10, 10, 1)},
        }
    }
]

output_lv = {
    'X': (0, 10.1, 0.1),
    'name': 'Junior Java skill level',
    'terms': {
        'none to very little': {'umf': ('trapmf', 0, 0, 0.59, 3.95), 'lmf': ('trapmf', 0, 0, 0.09, 1.32, 1)},
        'very low': {'umf': ('trapmf', 0.28, 2.00, 3.00, 5.22), 'lmf': ('trapmf', 1.79, 2.37, 2.37, 2.71, 0.48)},
        'low': {'umf': ('trapmf', 0.98, 2.75, 4.00, 5.41), 'lmf': ('trapmf', 2.79, 3.30, 3.30, 3.71, 0.42)},
        'medium': {'umf': ('trapmf', 2.38, 4.50, 6.00, 8.18), 'lmf': ('trapmf', 4.79, 5.12, 5.12, 5.35, 0.27)},
        'above medium': {'umf': ('trapmf', 4.02, 5.65, 7.00, 8.41), 'lmf': ('trapmf', 5.89, 6.34, 6.34, 6.81, 0.40)},
        'high': {'umf': ('trapmf', 4.38, 6.50, 7.75, 9.62), 'lmf': ('trapmf', 6.79, 7.25, 7.25, 7.91, 0.47)},
        'extremely high': {'umf': ('trapmf', 5.21, 8.27, 10, 10), 'lmf': ('trapmf', 7.66, 9.82, 10, 10, 1)},
    }
}

rule_base = [
    (('unskilled', 'unskilled', 'unskilled'), 'none to very little'),
    (('unskilled', 'unskilled', 'more or less unskilled'), 'none to very little'),
    (('unskilled', 'unskilled', 'more or less skilled'), 'very low'),
    (('unskilled', 'unskilled', 'highly skilled'), 'very low'),
    (('unskilled', 'more or less unskilled', 'unskilled'), 'none to very little'),
    (('unskilled', 'more or less unskilled', 'more or less unskilled'), 'very low'),
    (('unskilled', 'more or less unskilled', 'more or less skilled'), 'very low'),
    (('unskilled', 'more or less unskilled', 'highly skilled'), 'low'),
    (('unskilled', 'more or less skilled', 'unskilled'), 'very low'),
    (('unskilled', 'more or less skilled', 'more or less unskilled'), 'very low'),
    (('unskilled', 'more or less skilled', 'more or less skilled'), 'low'),
    (('unskilled', 'more or less skilled', 'highly skilled'), 'low'),
    (('unskilled', 'highly skilled', 'unskilled'), 'very low'),
    (('unskilled', 'highly skilled', 'more or less unskilled'), 'very low'),
    (('unskilled', 'highly skilled', 'more or less skilled'), 'low'),
    (('unskilled', 'highly skilled', 'highly skilled'), 'low'),
    (('more or less unskilled', 'unskilled', 'unskilled'), 'none to very little'),
    (('more or less unskilled', 'unskilled', 'more or less unskilled'), 'very low'),
    (('more or less unskilled', 'unskilled', 'more or less skilled'), 'low'),
    (('more or less unskilled', 'unskilled', 'highly skilled'), 'low'),
    (('more or less unskilled', 'more or less unskilled', 'unskilled'), 'very low'),
    (('more or less unskilled', 'more or less unskilled', 'more or less unskilled'), 'low'),
    (('more or less unskilled', 'more or less unskilled', 'more or less skilled'), 'low'),
    (('more or less unskilled', 'more or less unskilled', 'highly skilled'), 'low'),
    (('more or less unskilled', 'more or less skilled', 'unskilled'), 'low'),
    (('more or less unskilled', 'more or less skilled', 'more or less unskilled'), 'low'),
    (('more or less unskilled', 'more or less skilled', 'more or less skilled'), 'low'),
    (('more or less unskilled', 'more or less skilled', 'highly skilled'), 'medium'),
    (('more or less unskilled', 'highly skilled', 'unskilled'), 'low'),
    (('more or less unskilled', 'highly skilled', 'more or less unskilled'), 'low'),
    (('more or less unskilled', 'highly skilled', 'more or less skilled'), 'medium'),
    (('more or less unskilled', 'highly skilled', 'highly skilled'), 'medium'),
    (('more or less skilled', 'unskilled', 'unskilled'), 'very low'),
    (('more or less skilled', 'unskilled', 'more or less unskilled'), 'low'),
    (('more or less skilled', 'unskilled', 'more or less skilled'), 'medium'),
    (('more or less skilled', 'unskilled', 'highly skilled'), 'medium'),
    (('more or less skilled', 'more or less unskilled', 'unskilled'), 'low'),
    (('more or less skilled', 'more or less unskilled', 'more or less unskilled'), 'medium'),
    (('more or less skilled', 'more or less unskilled', 'more or less skilled'), 'medium'),
    (('more or less skilled', 'more or less unskilled', 'highly skilled'), 'above medium'),
    (('more or less skilled', 'more or less skilled', 'unskilled'), 'medium'),
    (('more or less skilled', 'more or less skilled', 'more or less unskilled'), 'medium'),
    (('more or less skilled', 'more or less skilled', 'more or less skilled'), 'above medium'),
    (('more or less skilled', 'more or less skilled', 'highly skilled'), 'above medium'),
    (('more or less skilled', 'highly skilled', 'unskilled'), 'medium'),
    (('more or less skilled', 'highly skilled', 'more or less unskilled'), 'medium'),
    (('more or less skilled', 'highly skilled', 'more or less skilled'), 'above medium'),
    (('more or less skilled', 'highly skilled', 'highly skilled'), 'high'),
    (('highly skilled', 'unskilled', 'unskilled'), 'low'),
    (('highly skilled', 'unskilled', 'more or less unskilled'), 'medium'),
    (('highly skilled', 'unskilled', 'more or less skilled'), 'medium'),
    (('highly skilled', 'unskilled', 'highly skilled'), 'above medium'),
    (('highly skilled', 'more or less unskilled', 'unskilled'), 'medium'),
    (('highly skilled', 'more or less unskilled', 'more or less unskilled'), 'above medium'),
    (('highly skilled', 'more or less unskilled', 'more or less skilled'), 'above medium'),
    (('highly skilled', 'more or less unskilled', 'highly skilled'), 'high'),
    (('highly skilled', 'more or less skilled', 'unskilled'), 'above medium'),
    (('highly skilled', 'more or less skilled', 'more or less unskilled'), 'above medium'),
    (('highly skilled', 'more or less skilled', 'more or less skilled'), 'high'),
    (('highly skilled', 'more or less skilled', 'highly skilled'), 'high'),
    (('highly skilled', 'highly skilled', 'unskilled'), 'above medium'),
    (('highly skilled', 'highly skilled', 'more or less unskilled'), 'high'),
    (('highly skilled', 'highly skilled', 'more or less skilled'), 'high'),
    (('highly skilled', 'highly skilled', 'highly skilled'), 'extremely high'),
]
