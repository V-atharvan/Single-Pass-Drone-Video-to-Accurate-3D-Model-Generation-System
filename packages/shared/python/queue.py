"""
SQS / Queue messaging wrapper for dispatching reconstruction jobs.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import boto3
from botocore.client import Config


class SQSQueueClient:
    """
    Publisher and consumer client for Amazon SQS.
    Enables asynchronous decoupling between FastAPI control plane and GPU workers.
    """

    def __init__(
        self,
        queue_url: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region_name: str = "us-east-1",
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ) -> None:
        self.queue_url = queue_url or os.environ.get(
            "SQS_QUEUE_URL", "http://localhost:4566/000000000000/reconstruction-jobs"
        )
        self._client = boto3.client(
            "sqs",
            endpoint_url=endpoint_url or os.environ.get("SQS_ENDPOINT_URL"),
            region_name=region_name,
            aws_access_key_id=access_key or os.environ.get("AWS_ACCESS_KEY_ID", "test"),
            aws_secret_access_key=secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
            config=Config(signature_version="s3v4"),
        )

    def publish_job(self, message: dict[str, Any], queue_url: Optional[str] = None) -> str:
        """Publish job payload to SQS queue and return MessageId."""
        target_url = queue_url or self.queue_url
        response = self._client.send_message(
            QueueUrl=target_url,
            MessageBody=json.dumps(message),
        )
        return response.get("MessageId", "")

    def receive_messages(
        self,
        max_messages: int = 1,
        wait_time_seconds: int = 20,
        queue_url: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Long-poll messages from SQS queue."""
        target_url = queue_url or self.queue_url
        response = self._client.receive_message(
            QueueUrl=target_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_time_seconds,
        )
        return response.get("Messages", [])

    def delete_message(self, receipt_handle: str, queue_url: Optional[str] = None) -> None:
        """Acknowledge and remove message from SQS queue."""
        target_url = queue_url or self.queue_url
        self._client.delete_message(
            QueueUrl=target_url,
            ReceiptHandle=receipt_handle,
        )
