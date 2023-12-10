import numpy as np
import json
import time

def sort_variables(variables):
    ## sort variables by x1, x2, x3, ...
    # get x index
    x_index = []
    for i in range(len(variables)):
        if 'x' in variables[i]:
            x_index.append(variables[i][variables[i].index('x')+1:])
    
    # sort variables
    if len(x_index) > 0:
        variables = [x for _,x in sorted(zip(x_index, variables))]
    
    return variables

def parse_coefficients(variables):
    ## get coefficients
    coefficients = []
    for variable in variables:
        ## cut off x and its index
        if 'x' in variable:
            variable = variable[:variable.index('x')]
        ## get coefficient
        if variable == '' or variable == '+':
            coefficients.append(1)
        elif variable == '-':
            coefficients.append(-1)
        else:
            coefficients.append(int(variable))
    return coefficients

def add_missing_variables(variables, max_x_index):
    ## add missing variables until xn
    if len(variables) == 0:
        return variables
    
    ## get current x indices
    current_x_indices = []
    for i in range(len(variables)):
        if 'x' in variables[i]:
            current_x_indices.append(variables[i][variables[i].index('x')+1:])
    
    ## get missing variables
    missing_variables = []
    for i in range(1, int(max_x_index)+1):
        if str(i) not in current_x_indices:
            missing_variables.append('0x' + str(i))
            
    ## add missing variables
    for variable in missing_variables:
        variables.append(variable)
    
    return variables

def get_max_x_index(variables):
    ## convert string to list
    variables = variables.split(' ')
    
    ## get x index
    x_index = []
    for i in range(len(variables)):
        if 'x' in variables[i]:
            x_index.append(variables[i][variables[i].index('x')+1:])
    
    if len(x_index) == 0:
        return False
    
    return max(x_index)

def flip_sign(variable):
    ## check if variable is integer or string
    if type(variable) == int:
        ## flip sign
        variable = -variable
    else:
        if '+' in variable:
            variable = variable.replace('+', '-')
        elif '-' in variable:
            variable = variable.replace('-', '+')
        else:
            variable = '+' + variable
    return variable

def rearrange_variables(variables_lh, variables_rh, constraint_sign, is_objective = False):
    if is_objective:
        ## put all variables to the right hand side
        delete_index = []
        for index, variable in enumerate(variables_lh):
            if 'z' not in variable:
                variable = flip_sign(variable)
                variables_rh.append(variable)
                delete_index.append(index)

        delete_index = sorted(delete_index, reverse=True)
        for index in delete_index:
            del variables_lh[index]
                
    
    if not is_objective:
        ## put all variables to the left hand side
        delete_index = []
        for index, variable in enumerate(variables_rh):
            if 'x' not in variable:
                continue
        
            variable = flip_sign(variable)
            variables_lh.append(variable)
            delete_index.append(index)
        
        delete_index = sorted(delete_index, reverse=True)
        for index in delete_index:
            del variables_rh[index]
        
        ## if negative solution, flip sign of solution
        summation = sum(int(i) for i in variables_rh)
        if summation < 0:
            variables_rh = [str(flip_sign(summation))]
            variables_lh = [flip_sign(i) for i in variables_lh] 
            if constraint_sign == '<=':
                constraint_sign = '>='
            elif constraint_sign == '>=':
                constraint_sign = '<='
        else:
            variables_rh = [str(summation)]
        
    return variables_lh, variables_rh, constraint_sign

def parse_variables(equation, max_x_index):
    variables_lh = []
    variables_rh = []
    
    constraint_sign = ''
    ## get variables and coefficients
    ## remove spaces
    equation = equation.replace(' ', '')
    
    if '<=' in equation:
        constraint_sign = '<='
    elif '>=' in equation:
        constraint_sign = '>='
    elif '=' in equation:
        constraint_sign = '='
        
    equation = equation.split(constraint_sign)
    
    ## get variables and coefficients on the left hand side
    variable = ''
    isSigned = False
    for i in range(len(equation[0])):
        if equation[0][i] in ['+', "-"] and not isSigned:
            if len(variable) > 0:
                variables_lh.append(variable)
            isSigned = True
            variable = equation[0][i]
        elif equation[0][i] in ['+', "-"] and isSigned:
            variables_lh.append(variable)
            variable = equation[0][i]
            isSigned = True
            continue
        elif equation[0][i] not in ['+', "-"]:
            variable += equation[0][i]
        
    variables_lh.append(variable)

    ## get variables and coefficients on the right hand side
    variable = ''
    isSigned = False
    for i in range(len(equation[1])):
        if equation[1][i] in ['+', "-"] and not isSigned:
            if len(variable) > 0:
                variables_rh.append(variable)
            isSigned = True
            variable = equation[1][i]
        elif equation[1][i] in ['+', "-"] and isSigned:
            variables_rh.append(variable)
            variable = equation[1][i]
            isSigned = True
            continue
        elif equation[1][i] not in ['+', "-"]:
            variable += equation[1][i]
            
    variables_rh.append(variable)
    
    ## check if objective function
    is_objective = False
    for variable in variables_lh:
        if 'z' in variable:
            is_objective = True
            break

    if not is_objective:
        for variable in variables_rh:
            if 'z' in variable:
                is_objective = True
                break
    
    ## for objective function put all variables to the right hand side and for the others put all variables to the left hand side
    variables_lh, variables_rh, constraint_sign = rearrange_variables(variables_lh, variables_rh, constraint_sign, is_objective)
    
    ## add missing variables
    if is_objective:
        variables_rh = add_missing_variables(variables_rh, max_x_index)
    else:
        variables_lh = add_missing_variables(variables_lh, max_x_index)
    
    ## sort variables
    variables_lh = sort_variables(variables_lh)
    variables_rh = sort_variables(variables_rh)

    
    ## get coefficients
    coefficients_lh = []
    if is_objective:
        coefficients_lh = [0]
    else:
        coefficients_lh = parse_coefficients(variables_lh)
    coefficients_rh = parse_coefficients(variables_rh)
    
    return {
        'variables_lh': variables_lh,
        'variables_rh': variables_rh,
        'coefficients_lh': coefficients_lh,
        'coefficients_rh': coefficients_rh,
        'constraint_sign': constraint_sign,
        'is_objective': is_objective
    }

def add_slack_variables(subject_to):
    n_slack_variables = 0
    n_surplus_variables = 0
    basic_variables = []
    slack_indices = []
    surplus_indices = []
    for index, constraint in enumerate(subject_to):
        if constraint['constraint_sign'] == '<=':
            constraint['variables_lh'].append('+s' + str(n_slack_variables+1))
            constraint['coefficients_lh'].append(1)
            constraint['constraint_sign'] = '='
            constraint['artificial'] = False
            basic_variables.append('s' + str(n_slack_variables+1))
            n_slack_variables += 1
            slack_indices.append(len(constraint['coefficients_lh'])-1)
            for index_j in range(len(subject_to)):
                if index_j != index:
                    subject_to[index_j]['coefficients_lh'].append(0)
        elif constraint['constraint_sign'] == '>=':
            constraint['variables_lh'].append('-s' + str(n_slack_variables+1))
            constraint['coefficients_lh'].append(-1)
            constraint['constraint_sign'] = '='
            constraint['artificial'] = True
            surplus_indices.append(len(constraint['coefficients_lh'])-1)
            n_surplus_variables += 1
            for index_j in range(len(subject_to)):
                if index_j != index:
                    subject_to[index_j]['coefficients_lh'].append(0)
        elif constraint['constraint_sign'] == '=':
            constraint['artificial'] = True
            
    return subject_to, n_slack_variables, basic_variables, slack_indices, surplus_indices, n_surplus_variables

def add_artificial_variables(subject_to):
    n_artificial_variables = 0
    artificial_variables = []
    artificial_indices = []
    for index, constraint in enumerate(subject_to):
        if constraint['artificial'] == True:
            constraint['variables_lh'].append('+a' + str(n_artificial_variables+1))
            constraint['coefficients_lh'].append(1)
            constraint['constraint_sign'] = '='
            artificial_variables.append('a' + str(n_artificial_variables+1))
            artificial_indices.append(len(constraint['coefficients_lh'])-1)
            for index_j in range(len(subject_to)):
                if index_j != index:
                    subject_to[index_j]['coefficients_lh'].append(0)

    return subject_to, artificial_indices, artificial_variables

def apply_problem_type(objective, p_type):
    if p_type == 'maximize':
        objective['coefficients_rh'] = [-i for i in objective['coefficients_rh']]
        objective['variables_rh'] = [flip_sign(i) for i in objective['variables_rh']]
        objective['variables_lh'] = ['f']
    else:
        objective['variables_lh'] = ['f']
    
    return objective

def construct_new_objective(subject_to, n_artificial_variables, revised = False):
    constraints = []
    non_artificial_indices = []
    for index, constraint in enumerate(subject_to):
        if constraint['artificial'] == True:
            constraints.append(subject_to[index].copy())
            continue
            
        if revised:
            constraint = subject_to[index].copy()
            constraint['coefficients_rh'] = [0]
            constraints.append(constraint)
            non_artificial_indices.append(index)
            continue
    
    ## construct new objective function
    coefficients_lh = []
    coefficients_rh = []
    
    for index, constraint in enumerate(constraints):
        if coefficients_lh == [] and coefficients_rh == []:
            coefficients_lh = constraint['coefficients_lh'].copy()
            coefficients_rh = constraint['coefficients_rh'].copy()
            continue

        for index_j, coefficient in enumerate(constraint['coefficients_lh']):
            coefficients_lh[index_j] += coefficient  
        
        for index_j, coefficient in enumerate(constraint['coefficients_rh']):
            coefficients_rh[index_j] += coefficient

    ## flip sign of coefficients
    coefficients_lh = [-i for i in coefficients_lh]
    coefficients_rh = [-i for i in coefficients_rh]
    
    ## replace artifical variables with 0
    for i in range(n_artificial_variables):
        coefficients_lh[-(i+1)] = 0
    
    
    return {
        'variables_lh': ['w'],
        'variables_rh': [''],
        'coefficients_lh': coefficients_lh,
        'coefficients_rh': coefficients_rh,
        'constraint_sign': '=',
        'is_objective': True
    }

def construct_simplex_table(objective, subject_to, w_objective = []):
    ## construct simplex table
    table = []

    ## add subject to constraints
    for constraint in subject_to:
        table.append(constraint['coefficients_lh'] + constraint['coefficients_rh'])
    
    if len(w_objective) > 0:
        table.append(w_objective['coefficients_lh'] + w_objective['coefficients_rh'])
    
    ## add objective function
    table.append(objective['coefficients_rh'] + objective['coefficients_lh'])

    return table

def print_table(table, title=''):
    ## print simplex table
    print(title)
    
    ## check if numpy array
    if isinstance(table, np.ndarray):
        table = table.tolist()
        
    for row in table:
        if isinstance(row, list):
            for value in row:
                ## round value to 4 decimal places
                value = round(value, 4)
                print('{:10}'.format(value), end=' ')
            print()
        else:
            ## round value to 4 decimal places
            row = round(row, 4)
            print('{:10}'.format(row))

def iterate_simplex(table, basic_variables, n_artificial_variables, phase = 1, artificial = False):
    ## print table
    print_table(table, 'Simplex Table')
    
    ## find objective function row
    if artificial:
        objective_row = len(table)-2
    else:
        objective_row = len(table)-1
    
    ## find the most negative value in the objective function
    most_negative_value = min(table[objective_row][:-1])
    most_negative_value_index = table[objective_row].index(most_negative_value)
    
    ## check if optimal solution is found
    if most_negative_value >= 0 and phase == 1 and artificial:
        # remove w_objective row
        for index, row in enumerate(table):
            if index == objective_row:
                continue
            
            ## remove artificial variables from each row
            table[index] = table[index][:-(n_artificial_variables+1)] + table[index][-1:]
        
        del table[objective_row]
        
        # remove w_objective from basic variables
        del basic_variables[-2]

        print ('Phase 1 completed')

        return iterate_simplex(table, basic_variables, n_artificial_variables, phase = 2, artificial = False)
    elif most_negative_value >= 0 and not artificial:
        print(f'phase {phase} completed')
        print('Optimal solution found')
        return table, basic_variables
    
    ## calculate ratios for each column
    ratios = []
    print('Most negative value: ', most_negative_value, ' at column ', most_negative_value_index+1)
    for index, row in enumerate(table):
        if index == objective_row or index == len(table)-1:
            continue
        if row[most_negative_value_index] <= 0:
            ratios.append(0)
            continue
        ratios.append(row[-1]/row[most_negative_value_index])
    
    ## find the smallest non-negative ratio
    if max(ratios) == 0:
        print('Unbounded solution')
        return False, False
    
    smallest_ratio = min([i for i in ratios if i > 0])
    smallest_ratio_index = ratios.index(smallest_ratio)
    
    ## divide pivot row by pivot value
    pivot_value = table[smallest_ratio_index][most_negative_value_index]
    table[smallest_ratio_index] = [(i/pivot_value) for i in table[smallest_ratio_index]]
    
    ## make pivot column 0
    for index, row in enumerate(table):
        if index == smallest_ratio_index:
            continue
        multiplier = row[most_negative_value_index]
        table[index] = [table[index][i] - multiplier*table[smallest_ratio_index][i] for i in range(len(row))]
        
    ## update basic variables
    basic_variables[smallest_ratio_index] = 'x' + str(most_negative_value_index+1)
    
    ## change small values to 0
    for index, row in enumerate(table):
        for index_j, value in enumerate(row):
            if abs(value) < 1e-5 and abs(value) > 0:
                table[index][index_j] = 0
    
    return iterate_simplex(table, basic_variables, n_artificial_variables, phase = phase, artificial = artificial)

def print_solution(solution_table, basic_variables, p_type):
    ## print solution
    print('Solution:')
    for index, variable in enumerate(basic_variables):
        if 'x' in variable:
            print(f'{variable} = {solution_table[index][-1]}')
            
    if solution_table[-1][-1] == 0:
        objective_solution = 0
    else:
        objective_solution = solution_table[-1][-1] if p_type == 'maximize' else -solution_table[-1][-1]
    
    print(f'f = {objective_solution}')

def inverse_matrix(matrix):
    ## inverse matrix
    return np.linalg.inv(matrix)

def multiply_matrix(matrix1, matrix2):
    ## multiply matrix
    return np.matmul(matrix1, matrix2)

def transpose_matrix(matrix):
    ## transpose matrix
    return np.transpose(matrix)

def row_operation(table, leaving_vector_index, multiplier_vector, entering_vector, artificial = False):
    ## update table
    ## divide pivot row by corresponding multiplier_vector value
    pivot_row = table[leaving_vector_index]
    multiplier_value = multiplier_vector[leaving_vector_index]
    pivot_row = [i/multiplier_value for i in pivot_row]
    table[leaving_vector_index] = pivot_row
    
    ## apply (new_row)+(-multiplier)*(pivot_row)
    for index, row in enumerate(table):
        ## if not pivot row put as 0 else put as 1
        entering_vector[index] = 0 if index != leaving_vector_index else 1
        
        if index == leaving_vector_index:
            continue
        
        multiplier = multiplier_vector[index]
        table[index] = [table[index][i] - multiplier*pivot_row[i] for i in range(len(row))]
        # if artificial:
        #     #entering_vector[index] = entering_vector[index] - multiplier*entering_vector[leaving_vector_index]
        #     entering_vector[index] = entering_vector[index] + table[index][leaving_vector_index]
        # else:
        #     entering_vector[index] = entering_vector[index] - multiplier*entering_vector[leaving_vector_index]
    return table, entering_vector

def construct_identity(matrix, basic_columns):
    ## rearrange matrix to identity matrix
    for col_index, col in enumerate(matrix):
        for row_index, row in enumerate(col):
            if col_index == row_index:
                if row == 1:
                    continue
                
                ## find the next col that has 1 in the same row
                for col_index_2, col_2 in enumerate(matrix):
                    if col_index_2 == col_index:
                        continue
                    if col_2[row_index] == 1:
                        ## swap col and col_2
                        matrix[:, [col_index, col_index_2]] = matrix[:, [col_index_2, col_index]]
                        basic_columns[col_index], basic_columns[col_index_2] = basic_columns[col_index_2], basic_columns[col_index]
                        break
                
                    
    return matrix, basic_columns
    
def iterate_revised_simplex(table, basic_columns, p_type, artificial = False, n=0, slack_indices = [], artificial_indices = []):
    timer_array = []
    start_time = time.time()
    print_table(table, f'Iteration {n}')
    
    ## get variables and coefficients vectors
    basic_variables_vector = []
    non_basic_variables_vector = []
    
    timer_array.append(time.time() - start_time)
    ## select basic variables vector
    basic_variables_vector = table[:, basic_columns]
    
    ## construct identity matrix out of basic variables vector
    #if n == 0:
    #    basic_variables_vector, basic_columns = construct_identity(basic_variables_vector, basic_columns)  
    
    inverse_basic_variables_vector = basic_variables_vector
    
    ## get non basic variables vector
    non_basic_columns = [i for i in range(len(table[0])-1) if i not in basic_columns and i != len(table[0])-1]
    non_basic_variables_vector = table[:, non_basic_columns]
    
    ## get solution vector
    solution_vector = table[:, -1]
    
    ## calculate new solution vector
    new_solution_vector = multiply_matrix(inverse_basic_variables_vector, solution_vector)
    
    if artificial:
        cost_step = 1
    else:
        cost_step = 0
    timer_array.append(time.time() - start_time)
    costs = multiply_matrix(inverse_basic_variables_vector[cost_step], non_basic_variables_vector)  
    
    ## check if optimal solution is found
    if min(costs) >= 0:
        if artificial:
            artificial = False
            ## remove artificial variables from table using indicies
            table = np.delete(table, artificial_indices, axis=1)
            
            ## remove artificial variables from basic columns
            for index, artificial_index in enumerate(artificial_indices):
                if artificial_index in basic_columns:
                    basic_columns.remove(artificial_index)
            
            ## recall function
            return iterate_revised_simplex(table, basic_columns, p_type, artificial = artificial, n=n+1, slack_indices = slack_indices, artificial_indices = artificial_indices)
        
        print('Optimal solution found')
        print('-------------------------')
        print('Solution:')
        for i, variable in enumerate(basic_columns):
            if artificial_indices != []:
                if i <= 1:
                    continue
                print(f'x{variable-1} = {solution_vector[i]}')
            else:
                if i <= 0:
                    continue
                print(f'x{variable} = {solution_vector[i]}')            
           
        if solution_vector[0] == 0:
            objective_solution = 0
        else:
            objective_solution = solution_vector[0] if p_type == 'maximize' else - solution_vector[0]
        print(f'f = {objective_solution}')
        return
    
    selected_value = min(costs)
    selected_value_index = np.where(costs == selected_value)[0][0]
    entering_vector_index = non_basic_columns[selected_value_index]
    entering_vector = non_basic_variables_vector[:, selected_value_index]
    
    ## calculate ratios for leaving vector
    multiplier_vector = multiply_matrix(inverse_basic_variables_vector, entering_vector)
    timer_array.append(time.time() - start_time)

    ratios = []
    for i in range(len(multiplier_vector)):
        if multiplier_vector[i] <= 0:
            ratios.append(0)
            continue
        ratios.append(solution_vector[i]/multiplier_vector[i])
    
    # find the smallest non-negative non-zero ratio
    positive_ratios = [ratio for ratio in ratios if ratio > 0]
    if len(positive_ratios) == 0:
        print('Unbounded solution')
        return
    smallest_ratio = min(positive_ratios)
    
    leaving_vector_index = ratios.index(smallest_ratio)
    
    leaving_vector = basic_variables_vector[:, leaving_vector_index]
    
    
    # ## get new basic variables vector
    # basic_variables_vector = table[:, basic_columns]
    timer_array.append(time.time() - start_time)

    
    ## apply row operation
    ## add solution vector to basic variables vector as the last column
    row_operation_table = np.insert(basic_variables_vector.copy(), len(basic_variables_vector[0]), solution_vector, axis=1)
    row_operation_table, new_leaving_vector = row_operation(row_operation_table, leaving_vector_index, multiplier_vector, entering_vector, artificial = artificial)
    
    ## update basic variables
    leaving_vector_table_index = basic_columns[leaving_vector_index]
    basic_columns[leaving_vector_index] = entering_vector_index
    
    ## update table
    new_basic_variables_vector = row_operation_table[:, :-1]
    table[:, basic_columns] = new_basic_variables_vector
    table[:, leaving_vector_table_index] = new_leaving_vector
    new_solution_vector = row_operation_table[:, -1]
    table[:, -1] = new_solution_vector

    timer_array.append(time.time() - start_time)
          
    ## print current step findings
    print('Current Step Findings')
    print('-------------------------')
    print_table(basic_variables_vector, 'Basic Variables Vector')
    print_table(inverse_basic_variables_vector.tolist(), 'Inverse Basic Variables Vector')
    print_table(entering_vector, 'Entering Vector')
    print_table(leaving_vector, 'Leaving Vector')
    print_table(multiplier_vector, 'Multiplier Vector')
    print('-------------------------')
    timer_array.append(time.time() - start_time)

    return iterate_revised_simplex(table, basic_columns, p_type, artificial = artificial, n=n+1, slack_indices = slack_indices, artificial_indices = artificial_indices)

def load_problems():
    with open('problems.json') as json_file:
        problems = json.load(json_file)
    return problems

def runner(objective, constraints, p_type):
    max_x_index = get_max_x_index(objective)
    for constraint in constraints:
        constraint_max_index = get_max_x_index(constraint)
        if constraint_max_index:
            if constraint_max_index > max_x_index:
                max_x_index = constraint_max_index
    
    objective_traditional = parse_variables(objective, max_x_index)
    objective_revised = parse_variables(objective, max_x_index)
    
    constraints = [parse_variables(constraint, max_x_index) for constraint in constraints]

    ## Start Simplex Algorithm
    constraints, n_slack_variables, basic_variables, slack_indices, surplus_indices, n_surplus_variables = add_slack_variables(constraints)
    constraints, artificial_indices, artificial_variables = add_artificial_variables(constraints)
    basic_variables = basic_variables + artificial_variables
    n_artificial_variables = len(artificial_variables)
    
    artifical = False
    if n_artificial_variables > 0:
        artifical = True
    
    ## apply equation type to objective function
    objective_traditional = apply_problem_type(objective_traditional, p_type)
    objective_revised = apply_problem_type(objective_revised, p_type)
    
    w_objective_traditional = []
    w_objective_revised = []
    ## construct new objective function
    if artifical:
        w_objective_traditional = construct_new_objective(constraints, n_artificial_variables)
        w_objective_revised = construct_new_objective(constraints, n_artificial_variables, revised = True)

    ## pad objective function with zeros
    for i in range(n_slack_variables + n_surplus_variables + n_artificial_variables):
        objective_traditional['coefficients_rh'].append(0)
        objective_revised['coefficients_rh'].append(0)

    if len(w_objective_traditional) > 0:
        basic_variables.append('w')
    
    basic_variables.append('f')
    
    ## construct simplex table
    simplex_table_traditional = construct_simplex_table(objective_traditional, constraints, w_objective = w_objective_traditional)
    simplex_table_revised = construct_simplex_table(objective_revised, constraints, w_objective = w_objective_revised)
    
    ## apply simplex algorithm
    solution_table, basic_variables = iterate_simplex(simplex_table_traditional, basic_variables, n_artificial_variables, phase = 1, artificial = artifical)
    
    if solution_table:
        print_solution(solution_table, basic_variables, p_type)


    time.sleep(3)
    ## apply revised simplex algorithm
    simplex_table_revised = np.array(simplex_table_revised, dtype = float)
    if artifical:
        ## move the last two rows to the first two rows
        simplex_table_revised = np.roll(simplex_table_revised, 2, axis=0)
        ## interchange the first two rows
        simplex_table_revised[[0, 1]] = simplex_table_revised[[1, 0]]
        omega = [0, 1]
        omega.extend([0 for i in range(len(simplex_table_revised)-2)])
        simplex_table_revised = np.insert(simplex_table_revised, 0, omega, axis=1)
    else:
        ## move the last row to the first row
        simplex_table_revised = np.roll(simplex_table_revised, 1, axis=0)
        
    ## add f as the first col
    f = [1]
    f.extend([0 for i in range(len(simplex_table_revised)-1)])
    simplex_table_revised = np.insert(simplex_table_revised, 0, f, axis=1)

    print('-------------------------')
    print('Revised Simplex Algorithm')
    print('-------------------------')
    
    
    ## construct basic variables
    basic_columns = []
    
    if artifical:
        slack_indices = [i+2 for i in slack_indices]
        artificial_indices = [i+2 for i in artificial_indices]
        basic_columns.extend(slack_indices)
        basic_columns.extend(artificial_indices)
        basic_columns.insert(0, 1)
        basic_columns.insert(0, 0)
    else:
        slack_indices = [i+1 for i in slack_indices]
        basic_columns.extend(slack_indices)
        basic_columns.insert(0, 0)

    #print_table(simplex_table_revised, 'Revised Simplex Table')
    
    ## apply revised simplex algorithm
    iterate_revised_simplex(simplex_table_revised, basic_columns, p_type, artificial = artifical, slack_indices = slack_indices, artificial_indices = artificial_indices)
    
    time.sleep(5)

def main():
    ## basic variables are assumed to be non-negative
    problems = load_problems()
        
    for problem in problems:
        ## print problem
        print('Problem: ', problem['objective'])
        for constraint in problem['constraints']:
            print('Constraint: ', constraint)
        print('Problem Type: ', problem['problem_type'])
        print('-------------------------')
        time.sleep(3)
        runner(problem['objective'], problem['constraints'], problem['problem_type'])
        print('-------------------------')   

if __name__ == "__main__":
    main()