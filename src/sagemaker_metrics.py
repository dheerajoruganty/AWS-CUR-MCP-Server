"""
Retrieves metrics for SageMaker Endpoints from CloudWatch.
See https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-cloudwatch.html for
full list of metrics.
"""
import json
import boto3
import logging
import pandas as pd
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import List, Optional

# Setup logging
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models
class EndpointMetricParams(BaseModel):
    endpoint_name: str
    variant_name: str
    start_time: datetime
    end_time: datetime
    period: int = 60

class UtilizationDatapoint(BaseModel):
    Timestamp: datetime
    Average: float

class InvocationDatapoint(BaseModel):
    Timestamp: datetime
    Value: float

# Pydantic models for the structure of data before DataFrame creation
class UtilizationMetricRecord(BaseModel):
    EndpointName: str
    Timestamp: datetime
    MetricName: str
    Average: float

class InvocationMetricRecord(BaseModel):
    EndpointName: str
    Timestamp: datetime
    MetricName: str
    Value: float


def _get_endpoint_utilization_metrics(endpoint_name: str,
                                      variant_name: str,
                                      start_time: datetime,
                                      end_time: datetime,
                                      period : int = 60) -> pd.DataFrame:
    """
    Retrieves utilization metrics for a specified SageMaker endpoint within a given time range.

    Parameters:
    - endpoint_name (str): The name of the SageMaker endpoint.
    - variant_name (str): The name of the endpoint variant.
    - start_time (datetime): The start time for the metrics data.
    - end_time (datetime): The end time for the metrics data.
    - period (int): The granularity, in seconds, of the returned data points. Default is 60 seconds.

    Returns:
    - Dataframe: A Dataframe containing metric values for utilization metrics like CPU and GPU Usage.
    """
    
    metrics = ["CPUUtilization",
               "MemoryUtilization",
               "DiskUtilization",
               "InferenceLatency",
               "GPUUtilization",
               "GPUMemoryUtilization"]
    
    client = boto3.client('cloudwatch')
    data = []
    namespace = "/aws/sagemaker/Endpoints"
    
    for metric_name in metrics:
        logger.debug(f"_get_endpoint_utilization_metrics, endpoint_name={endpoint_name}, variant_name={variant_name}, "
                     f"metric_name={metric_name}, start_time={start_time}, end_time={end_time}")
        response = client.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric_name,
            Dimensions=[
                {
                    'Name': 'EndpointName',
                    'Value': endpoint_name
                },
                {
                    'Name': 'VariantName',
                    'Value': variant_name
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=period,
            Statistics=['Average']  # You can also use 'Sum', 'Minimum', 'Maximum', 'SampleCount'
        )
        logger.debug(response)
        for datapoint_raw in response['Datapoints']:
            # Validate with Pydantic before adding
            datapoint = UtilizationDatapoint(**datapoint_raw)
            data.append(UtilizationMetricRecord(
                EndpointName=endpoint_name, 
                Timestamp=datapoint.Timestamp,
                MetricName=metric_name,
                Average=datapoint.Average
            ).dict()) # Convert back to dict for DataFrame creation

    # Create a DataFrame from the collected data
    if not data:
        logger.warning(f"No utilization datapoints found for {endpoint_name} / {variant_name}")
        # Return empty DataFrame with expected columns if no data
        return pd.DataFrame(columns=['Timestamp', 'EndpointName'] + metrics)
        
    df = pd.DataFrame(data)

    # Pivot the DataFrame to have metrics as columns
    df_pivot = df.pivot_table(index=['Timestamp', 'EndpointName'], columns='MetricName', values='Average').reset_index()
    
    # Remove the index column heading
    sm_utilization_metrics_df = df_pivot.rename_axis(None, axis=1)
    
    return sm_utilization_metrics_df


def _get_endpoint_invocation_metrics(endpoint_name: str,
                                     variant_name: str,
                                     start_time: datetime,
                                     end_time: datetime,
                                     period : int = 60):
    """
    Retrieves Invocation metrics for a specified SageMaker endpoint within a given time range.

    Parameters:
    - endpoint_name (str): The name of the SageMaker endpoint.
    - variant_name (str): The name of the endpoint variant.
    - start_time (datetime): The start time for the metrics data.
    - end_time (datetime): The end time for the metrics data.
    - period (int): The granularity, in seconds, of the returned data points. Default is 60 seconds.

    Returns:
    - Dataframe: A Dataframe containing metric values for Invocation metrics like Invocations and Model Latency.
    """
    metric_names = ["Invocations",
                    "Invocation4XXErrors",
                    "Invocation5XXErrors",
                    "ModelLatency",
                    "InvocationsPerInstance"]
    
    # Initialize a session using Amazon CloudWatch
    client = boto3.client('cloudwatch')

    namespace = "AWS/SageMaker"
    data = []
    
    for metric_name in metric_names:
        if metric_name == 'ModelLatency':
            stat = 'Average'
        else:
            stat = 'Sum'
        logger.debug(f"_get_endpoint_invocation_metrics, endpoint_name={endpoint_name}, variant_name={variant_name}, "
                     f"metric_name={metric_name}, start_time={start_time}, end_time={end_time}")
        # Get metric data for the specified metric
        response = client.get_metric_data(
            MetricDataQueries=[
                {
                    'Id': f'metric_{metric_name}',
                    'MetricStat': {
                        'Metric': {
                            'Namespace': namespace,
                            'MetricName': metric_name,
                            'Dimensions': [
                                {
                                    'Name': 'EndpointName',
                                    'Value': endpoint_name
                                },
                                {
                                    'Name': 'VariantName',
                                    'Value': variant_name
                                }
                            ]
                        },
                        'Period': period,  # Period in seconds
                        'Stat': stat  # Statistic to retrieve
                    },
                    'ReturnData': True,
                },
            ],
            StartTime=start_time,
            EndTime=end_time
        )
        logger.debug(response)
        # Extract the data points from the response
        timestamps = response['MetricDataResults'][0]['Timestamps']
        values = response['MetricDataResults'][0]['Values']
        
        for timestamp_raw, value_raw in zip(timestamps, values):
            # Validate with Pydantic
            datapoint = InvocationDatapoint(Timestamp=timestamp_raw, Value=value_raw)
            data.append(InvocationMetricRecord(
                EndpointName=endpoint_name, 
                Timestamp=datapoint.Timestamp,
                MetricName=metric_name,
                Value=datapoint.Value
            ).dict()) # Convert back to dict for DataFrame creation

    # Create a DataFrame from the collected data
    if not data:
        logger.warning(f"No invocation datapoints found for {endpoint_name} / {variant_name}")
        # Return empty DataFrame with expected columns if no data
        return pd.DataFrame(columns=['Timestamp', 'EndpointName'] + metric_names)
        
    df = pd.DataFrame(data)
    
    # Pivot the DataFrame to have metrics as columns
    df_pivot = df.pivot_table(index=['Timestamp', 'EndpointName'], columns='MetricName', values='Value').reset_index()
    
    # Remove the index column heading
    sm_invocation_metrics_df = df_pivot.rename_axis(None, axis=1)
    
    return sm_invocation_metrics_df


def get_endpoint_metrics(params: EndpointMetricParams) -> Optional[pd.DataFrame]:
    """
    Retrieves Invocation and Utilization metrics for a specified SageMaker endpoint within a given time range.

    Parameters:
    - params (EndpointMetricParams): Pydantic model containing endpoint name, variant name, start time, end time, and period.

    Returns:
    - Optional[Dataframe]: A Dataframe containing metric values for Utilization and Invocation metrics, or None if an error occurs.
    """
    
    endpoint_metrics_df: Optional[pd.DataFrame] = None
    try:
        logger.info(f"get_endpoint_metrics, going to retrieve endpoint utlization metrics for "
                    f"endpoint={params.endpoint_name}, variant_name={params.variant_name}, start_time={params.start_time}, "
                    f"end_time={params.end_time}, period={params.period}")
        utilization_metrics_df = _get_endpoint_utilization_metrics(endpoint_name=params.endpoint_name,
                                                                   variant_name=params.variant_name,
                                                                   start_time=params.start_time,
                                                                   end_time=params.end_time,
                                                                   period=params.period)
        logger.info(f"get_endpoint_metrics, going to retrieve endpoint invocation metrics for "
                    f"endpoint={params.endpoint_name}, variant_name={params.variant_name}, start_time={params.start_time}, "
                    f"end_time={params.end_time}, period={params.period}")
        invocation_metrics_df = _get_endpoint_invocation_metrics(endpoint_name=params.endpoint_name,
                                                                 variant_name=params.variant_name,
                                                                 start_time=params.start_time,
                                                                 end_time=params.end_time,
                                                                 period=params.period)

        # Handle cases where one or both dataframes might be empty
        if utilization_metrics_df.empty and invocation_metrics_df.empty:
             logger.warning(f"No utilization or invocation metrics found for endpoint={params.endpoint_name}")
             return pd.DataFrame() # Return empty dataframe
        elif utilization_metrics_df.empty:
            logger.warning(f"No utilization metrics found for endpoint={params.endpoint_name}, returning only invocation metrics.")
            endpoint_metrics_df = invocation_metrics_df
        elif invocation_metrics_df.empty:
            logger.warning(f"No invocation metrics found for endpoint={params.endpoint_name}, returning only utilization metrics.")
            endpoint_metrics_df = utilization_metrics_df
        else:
            # Merge only if both have data
            endpoint_metrics_df = pd.merge(utilization_metrics_df,
                                        invocation_metrics_df,
                                        on=['Timestamp', 'EndpointName'],
                                        how='outer')
        
        if endpoint_metrics_df is not None and not endpoint_metrics_df.empty:
            logger.info(f"get_endpoint_metrics, shape of final metrics for "
                        f"endpoint={params.endpoint_name} is {endpoint_metrics_df.shape}")
            logger.info(f"get_endpoint_metrics, endpoint_metrics_df=\n{endpoint_metrics_df.head()}")
        else:
             logger.info(f"get_endpoint_metrics, final dataframe is empty or None for endpoint={params.endpoint_name}")
             
    except Exception as e:
        logger.error(f"get_endpoint_metrics, exception occured while retrieving metrics for {params.endpoint_name}, "
                     f"exception={e}")
        # In case of exception, ensure None is returned as per type hint
        return None 

    return endpoint_metrics_df