import torch
import flwr as fl
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from collections import OrderedDict

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            state_dict = OrderedDict({f'layer_{i}': torch.tensor(layer) for i, layer in enumerate(aggregated_ndarrays)})

            print(f"Saving round {server_round} aggregated_parameters...")
            torch.save(state_dict, f"round-{server_round}-weights.ckpt")

        return aggregated_parameters, aggregated_metrics


def main() -> None:
    print("creating strategy")
    strategy = SaveModelStrategy(
        fraction_fit=0.5, 
        fraction_evaluate=0.5, 
    )
    if strategy is not None:
        print("strategy initialized")

    print("server starting now at 8085")
    fl.server.start_server(
        server_address="0.0.0.0:8085",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
    print("global checkpoint saved")


if __name__ == "__main__":
    main()
