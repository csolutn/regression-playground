# regression-playground

A web GUI for exploring supervised regression: define data, pick a model, and watch training (prediction → loss → optimization) step by step.

## Planned

- Data: a target function (1 or 2 input variables → 1 output) or a CSV upload
- Models: neural network (choice of activation, layers, optimizer, batch method), linear regression, decision-tree family
- Live view of the fitted curve/surface and train/validation loss

## Run locally

```bash
uv sync
cp .env.example .env
uv run flask --app app run --debug
```
