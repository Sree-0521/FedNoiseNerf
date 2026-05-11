import flwr as fl


def main() -> None:
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5, #0.0
    )

    # strategy = fl.server.strategy.FedAvg(
    #     fraction_fit=1.0, #0.5
    #     fraction_evaluate=0.0, #0.0
    # )

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=fl.server.ServerConfig(num_rounds=2), #change
        strategy=strategy,
    )


if __name__ == "__main__":
    main()