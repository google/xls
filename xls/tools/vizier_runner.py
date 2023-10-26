from vizier import service
from vizier.service import clients
from vizier.service import pyvizier as vz
from vizier.service.pyvizier import SearchSpaceSelector, MetricInformation, ObjectiveMetricGoal, MetricsConfig
from vizier.service import servers
from xls.tools.dashboard.run_and_parse import BazelLabel, SystemPath, parse_file
from pathlib import Path
import argparse
import subprocess
import json
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Parameter:
    """
    Data type used for wrapping XLS parameter into optimization parameter.

    DSLX design parameters and implementation tool parameters can be specified
    as instances of this class and then used in generic functions for integrating
    those into optimization engine.

    Attributes
    ----------
    type : type
        Type of the parameter
    name : str
        Name of the parameter. Will be used by optimization engine and to construct
        environment variables for passing the parameters to bazel rules
    min_value: int or float
    max_value: int or float
        Constraints for the parameter. Will reduce the size of the search space.
    """
    type: type
    name: str
    min_value: int | float
    max_value: int | float

@dataclass
class TestData:
    """
    Data type used for storing data required for running tests which evaluate
    parameter values.

    Attributes
    ----------
    label : BazelLabel
        Bazel Label passed in string format which points to a python binary with
        test capable of obtaining performance metrics from implemented DSLX design.
    parser : SystemPath or str
        Path to parser responsible for extracting the performance metrics from test output.
    log : SystemPath or str
        Path to temporary log file used for storing test output for parsing.
    """
    label: BazelLabel
    parser: SystemPath | str
    log: SystemPath | str

def construct_parameters() -> List[Parameter]:
    """Specify a list of parameters to optimize and constraints for the search space."""
    parameters = []

    parameters.append(Parameter(name="PIPELINE_STAGES", type=int, min_value=1, max_value=5))
    parameters.append(Parameter(name="CLOCK_PERIOD", type=int, min_value=1, max_value=100))
    parameters.append(Parameter(name="TARGET_DIE_UTILIZATION_PERCENTAGE", type=int, min_value=0, max_value=100))
    parameters.append(Parameter(name="PLACEMENT_DENSITY", type=float, min_value=0.0, max_value=1.0))

    return parameters

def construct_performance_metrics() -> List[MetricInformation]:
    """Specify a list of performance metrics used in parameter optmization process"""
    metrics = []

    metrics.append(MetricInformation(name='chip_area', goal=ObjectiveMetricGoal.MINIMIZE))
    metrics.append(MetricInformation(name='delay', goal=ObjectiveMetricGoal.MINIMIZE))
    metrics.append(MetricInformation(name='throughput', goal=ObjectiveMetricGoal.MAXIMIZE))

    return metrics

def add_performance_metrics(metrics: List[MetricInformation], metric_information: MetricsConfig) -> None:
    """Configure the optimization engine with given performance metrics"""
    metric_information.extend(metrics)

def add_search_space_parameters(parameters: List[Parameter], search_space_root: SearchSpaceSelector) -> None:
    """Configure the parameters to be optimized by the optimization engine"""
    for param in parameters:
        if (param.type == int):
            search_space_root.add_int_param(name=param.name, min_value=param.min_value, max_value=param.max_value)
        elif (param.type == float):
            search_space_root.add_float_param(name=param.name, min_value=param.min_value, max_value=param.max_value)
        else:
            raise ValueError("Unsupported parameter type: {}".format(param.type))

def prepare_env(env) -> None:
    """Explicitly cast values of integer parameters to int.

    This is required because the optimization engine suggest float values even
    for parameters which were defined as int parameters in `construct_performance_metrics()`."""
    for var, value in env.items():
        if (var != "PLACEMENT_DENSITY"):
            env[var] = int(value)

def construct_bazel_command(test: BazelLabel, test_log: SystemPath, env) -> str:
    """Specify command for implementing and testing DSLX design

    Create two separate bazel commands, one for build, second for running tests
    and chain those into single command.
    Each bazel call should pass the parameter values as environmet variables
    to bazel build system through `--action-env` argument.
    Write test output to the `test_log`.
    """
    # Force the usage of custom optimization platform with constraint_value
    # `has_env_vars` defined which is required for running generic
    # implementation and testing rules generated with
    # generate_compression_block_parameter_optimization() macro
    force_platform = "--platforms //xls/build_rules:parameter_optimization_platform"
    build_cmd = f"bazel build {force_platform}"
    test_cmd = f"bazel test {force_platform}"
    for var, value in env.items():
        set_env_var = " --action_env " + var + "=" + str(value)
        build_cmd += set_env_var
        test_cmd += set_env_var
    return f"{build_cmd} {test} && {test_cmd} {test} --test_output=all > {test_log}"

def implement_and_test_design(cmd: str, cwd: SystemPath) -> None:
    """Run command line for implementing and testing DSLX design"""
    try:
        subprocess.check_call(cmd, shell=True, cwd=cwd)
    except subprocess.CalledProcessError:
        raise Exception(
           f"Error while executing Bazel test with the follwing command:\n{cmd}"
        )

def get_objective(test: TestData, root_dir) -> Dict:
    """Fetch performance metrics into structure consumable for the optimization engine."""
    results = parse_file(test.parser, test.log, root_dir)
    json_obj = json.loads(results)

    # Extract performance metrics from dashboard JSON
    delay = json_obj[0]["value"][1][1]
    throughput = json_obj[0]["value"][2][1]

    objective = {
        'chip_area': 0,
        'delay': delay,
        'throughput': throughput
    }

    return objective

def evaluate(env, root_dir, test: TestData) -> Dict:
    """Use suggested parameter values to obtain performance metrics
    """
    prepare_env(env) # Explicitly cast int parameters to integer type - this is probably Vizier bug, those parameters should be casted before being suggested
    bazel_command = construct_bazel_command(test.label, test.log, env)
    implement_and_test_design(bazel_command, root_dir)

    return get_objective(test, root_dir)

def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-r", "--root-directory", type=str, help="Root directory", required=True)
    parser.add_argument("-t", "--test-label", type=str, help="Bazel label string for running tests", required=True)
    parser.add_argument("-p", "--parser", type=str, help="Test output parser", required=True)
    parser.add_argument("-l", "--log-file", type=str, help="Path to log file for the test output", default="/tmp/test_log.txt")
    parser.add_argument("-i", "--iterations", type=int, help="Max iteration count for the optimization engine", default=5)
    parser.add_argument("-s", "--suggestions", type=int, help="Suggestion count per iteration of the optimization engine", default=1)
    parser.add_argument("-a", "--algorithm", type=str, choices=['GAUSSIAN_PROCESS_BANDIT', 'RANDOM_SEARCH', 'QUASI_RANDOM_SEARCH', 'GRID_SEARCH', 'SHUFFLED_GRID_SEARCH', 'EAGLE_STRATEGY', 'NSGA2', 'BOCS', 'HARMONICA'], help="Algorithm for the optimization engine", default='NSGA2')

    return parser.parse_args()

def check_args(args: argparse.Namespace) -> None:
    """Validate arguments passed to the parameter optimization script."""
    root_dir = Path(args.root_directory)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Root directory {str(root_dir)} does not exist")

    parser_path = root_dir / args.parser
    if not parser_path.exists():
        raise FileNotFoundError(f"Output parser {str(parser_path)} does not exist")

    if (args.iterations < 1):
        raise ValueError(f"Iteration count must be above 0 (got {str(args.iterations)})")

    if (args.suggestions < 1):
        raise ValueError(f"Suggestions count must be above 0 (got {str(args.suggestions)})")

def client_runner(client: 'Study', test_data: TestData, root_dir, iteration_cnt: int, suggestion_cnt: int) -> None:
    """
    Vizier Study Client performs parameter optimization.

    Run `iteration_cnt` iterations in which `suggestion_cnt` suggestions are generated.
    Each suggestion is a set of parameter values which are then evaluated for
    the performance metrics.
    """
    for i in range(iteration_cnt):
        suggestions = client.suggest(count=suggestion_cnt)
        for s, suggestion in enumerate(suggestions):
            objective = evaluate(suggestion.parameters, root_dir, test_data)
            print(f'Iteration {i}, suggestion {s} with parameters: {suggestion.parameters} led to objective value: {objective}.')
            final_measurement = vz.Measurement(objective)
            suggestion.complete(final_measurement)
    client.set_state(vz.StudyState.COMPLETED)

def print_client_optimal_trials(client: 'Study'):
    """Print information gathered by the client in optimization process."""

    if (client.materialize_state() != vz.StudyState.COMPLETED):
        raise RuntimeError(f"Client {client.resource_name} status is not COMPLETED yet")

    for optimal_trial in client.optimal_trials():
        optimal_trial = optimal_trial.materialize()
        print(f"Client {client.resource_name} Optimal Trial Suggestion: {optimal_trial.parameters} and Objective: {optimal_trial.final_measurement}")

def main() -> None:
    args = setup_args()
    check_args(args)

    root_dir = Path(args.root_directory)
    parser = root_dir / args.parser
    test_data = TestData(args.test_label, parser, args.log_file)

    print(f"Setting Optimization engine with parameters: {args}")
    problem = vz.ProblemStatement()

    parameters = construct_parameters()
    add_search_space_parameters(parameters, problem.search_space.root)

    performance_metrics = construct_performance_metrics()
    add_performance_metrics(performance_metrics, problem.metric_information)

    study_config = vz.StudyConfig.from_problem(problem)
    study_config.algorithm = args.algorithm

    # Vizier Server setup
    server_host = "localhost"
    server_database = service.SQL_LOCAL_URL
    server = servers.DefaultVizierServer(host=server_host, database_url=server_database)
    print(f"Started Vizier Server at: {server.endpoint}")
    print(f"SQL database file located at: {server._database_url}")

    # Vizier Client setup
    clients.environment_variables.server_endpoint = server.endpoint
    study_client = clients.Study.from_study_config(study_config, owner='owner', study_id='example_study_id')

    client_runner(study_client, test_data, root_dir, args.iterations, args.suggestions)
    print_client_optimal_trials(study_client)

if __name__ == "__main__":
    main()


