import numpy as np
import argparse
import os

def gaussian_cross_gamma(mean, stddev, shape, rate, n_samples_lambda):
    n_samples = np.random.poisson(n_samples_lambda)
    result = np.zeros((n_samples, 2))
    result[:, 0] = np.abs(np.random.normal(mean, stddev, size=n_samples))
    result[:, 1] = np.random.gamma(shape, rate, size=n_samples)
    return result
    

def create_diagram(gaussian_params=[(1.5, 0.5)], gamma_params=[(1.5, 0.5)], n_samples=[500]):

    components = [
        gaussian_cross_gamma(mean, stddev, shape, rate, n)
        for (mean, stddev), (shape, rate), n in zip(gaussian_params, gamma_params, n_samples)
    ]
    
    return np.vstack(components)
    

def lifetime_to_deathtime(diagram):
    result = diagram.copy()
    result[:, 1] = diagram.sum(axis=1)
    return result
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate many fake persistence diagrams')
    parser.add_argument('-d', '--directory_base', type=str, default='data', help="Basename for the output directory")
    parser.add_argument('-c', '--classes', type=str, default='1,2,3,4,5', help="Comma separated list of diagram classes to use")
    parser.add_argument('-n', '--n_points', type=int, default=50, help="Approximate number of points per diagram")
    parser.add_argument('-N', '--n_diagrams', type=int, default=100, help="Number of diagrams to generate per class")
    
    args = parser.parse_args()
    
    if not os.path.exists(f"{args.directory_base}_{args.n_points}"):
        os.mkdir(f"{args.directory_base}_{args.n_points}")
        
    base_n_samples = args.n_points
    n_diagrams_per_class = args.n_diagrams
    classes_to_generate = [int(c) for c in args.classes.split(",")]
        
    class1_params = dict(
        gaussian_params=[(1.5, 0.5)], 
        gamma_params=[(1.5, 0.5)], 
        n_samples=[base_n_samples]
    )
    
    class2_params = dict(
        gaussian_params=[(1.3, 0.5), (1.5, 0.1)], 
        gamma_params=[(1.5, 0.4), (15.0, 0.3)], 
        n_samples=[int(base_n_samples * 0.9), int(base_n_samples * 0.1)]
    )
    
    class3_params = dict(
        gaussian_params=[(1.5, 0.75), (2.0, 0.1)], 
        gamma_params=[(1.4, 0.2), (13.0, 0.25)], 
        n_samples=[int(base_n_samples * 0.9), int(base_n_samples * 0.1)]
    )
    
    class4_params = dict(
        gaussian_params=[(1.5, 0.75), (1.0, 0.1), (2.5, 0.1)], 
        gamma_params=[(1.4, 0.2), (16.0, 0.2), (20.0, 0.15)], 
        n_samples=[int(base_n_samples * 0.8), int(base_n_samples * 0.15), int(base_n_samples * 0.05)]
    )
        
    class5_params = dict(
        gaussian_params=[(1.9, 1.25), (0.5, 0.1), (3.0, 0.1)], 
        gamma_params=[(1.4, 0.2), (20.0, 0.2), (25.0, 0.2)], 
        n_samples=[int(base_n_samples * 0.8), int(base_n_samples * 0.1), int(base_n_samples * 0.1)]
    )
    
    for i in range(n_diagrams_per_class):
        if 1 in classes_to_generate:
            c1_diagram = lifetime_to_deathtime(create_diagram(**class1_params))
            np.savetxt(f"{args.directory_base}_{args.n_points}/class1_diagram_{i}.txt", c1_diagram)
        if 2 in classes_to_generate:
            c2_diagram = lifetime_to_deathtime(create_diagram(**class2_params))
            np.savetxt(f"{args.directory_base}_{args.n_points}/class2_diagram_{i}.txt", c2_diagram)
        if 3 in classes_to_generate:
            c3_diagram = lifetime_to_deathtime(create_diagram(**class3_params))
            np.savetxt(f"{args.directory_base}_{args.n_points}/class3_diagram_{i}.txt", c3_diagram)
        if 4 in classes_to_generate:
            c4_diagram = lifetime_to_deathtime(create_diagram(**class4_params))
            np.savetxt(f"{args.directory_base}_{args.n_points}/class4_diagram_{i}.txt", c4_diagram)
        if 5 in classes_to_generate:
            c5_diagram = lifetime_to_deathtime(create_diagram(**class5_params))
            np.savetxt(f"{args.directory_base}_{args.n_points}/class5_diagram_{i}.txt", c5_diagram)

    
