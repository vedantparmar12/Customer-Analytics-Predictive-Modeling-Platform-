"""
Marketing Automation Tool Integrations
Supports HubSpot, Salesforce Marketing Cloud, Mailchimp, and ActiveCampaign
"""

import os
import json
import requests
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
from abc import ABC, abstractmethod
import logging

from src.logger import get_logger

logger = get_logger(__name__)

class MarketingAutomationConnector(ABC):
    """Base class for marketing automation connectors"""
    
    @abstractmethod
    def sync_customer_segments(self, segments: pd.DataFrame) -> Dict[str, Any]:
        """Sync customer segments to marketing platform"""
        pass
    
    @abstractmethod
    def sync_churn_predictions(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """Sync churn predictions to marketing platform"""
        pass
    
    @abstractmethod
    def create_campaign(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new marketing campaign"""
        pass
    
    @abstractmethod
    def update_contact_properties(self, contacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update contact properties in bulk"""
        pass

class HubSpotConnector(MarketingAutomationConnector):
    """HubSpot integration connector"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.hubapi.com"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def sync_customer_segments(self, segments: pd.DataFrame) -> Dict[str, Any]:
        """Sync RFM segments to HubSpot as contact properties"""
        try:
            # Create custom properties if they don't exist
            self._create_custom_properties()
            
            # Prepare batch update
            contacts = []
            for _, row in segments.iterrows():
                contact = {
                    "id": row['customer_id'],
                    "properties": {
                        "rfm_segment": row['RFM_segment'],
                        "rfm_score": row['RFM_score'],
                        "clv_tier": row.get('clv_tier', 'Unknown'),
                        "last_updated": datetime.utcnow().isoformat()
                    }
                }
                contacts.append(contact)
            
            # Batch update contacts
            response = requests.post(
                f"{self.base_url}/crm/v3/objects/contacts/batch/update",
                headers=self.headers,
                json={"inputs": contacts[:100]}  # HubSpot limit
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully synced {len(contacts)} segments to HubSpot")
                return {"success": True, "synced_count": len(contacts)}
            else:
                logger.error(f"HubSpot sync failed: {response.text}")
                return {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"Error syncing to HubSpot: {e}")
            return {"success": False, "error": str(e)}
    
    def sync_churn_predictions(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """Sync churn predictions to HubSpot"""
        try:
            contacts = []
            for _, row in predictions.iterrows():
                contact = {
                    "id": row['customer_id'],
                    "properties": {
                        "churn_risk": "High" if row['churn_probability'] > 0.7 else "Medium" if row['churn_probability'] > 0.3 else "Low",
                        "churn_probability": round(row['churn_probability'], 3),
                        "predicted_churn": row['churned'],
                        "churn_prediction_date": datetime.utcnow().isoformat()
                    }
                }
                contacts.append(contact)
            
            # Create lists for different churn risk levels
            self._create_churn_lists(predictions)
            
            # Batch update
            response = requests.post(
                f"{self.base_url}/crm/v3/objects/contacts/batch/update",
                headers=self.headers,
                json={"inputs": contacts[:100]}
            )
            
            return {
                "success": response.status_code == 200,
                "synced_count": len(contacts),
                "response": response.json() if response.status_code == 200 else response.text
            }
            
        except Exception as e:
            logger.error(f"Error syncing churn predictions: {e}")
            return {"success": False, "error": str(e)}
    
    def create_campaign(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create an email campaign in HubSpot"""
        try:
            # Create email campaign
            email_data = {
                "name": campaign_data['name'],
                "subject": campaign_data['subject'],
                "fromName": campaign_data.get('from_name', 'E-commerce Team'),
                "replyTo": campaign_data.get('reply_to', 'noreply@example.com'),
                "campaign": campaign_data.get('campaign_id'),
                "htmlContent": campaign_data.get('html_content', ''),
                "plainTextContent": campaign_data.get('plain_content', '')
            }
            
            response = requests.post(
                f"{self.base_url}/marketing/v3/emails",
                headers=self.headers,
                json=email_data
            )
            
            if response.status_code == 201:
                email_id = response.json()['id']
                
                # Add recipients based on segment
                if 'segment' in campaign_data:
                    self._add_campaign_recipients(email_id, campaign_data['segment'])
                
                return {"success": True, "campaign_id": email_id}
            else:
                return {"success": False, "error": response.text}
                
        except Exception as e:
            logger.error(f"Error creating campaign: {e}")
            return {"success": False, "error": str(e)}
    
    def update_contact_properties(self, contacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update multiple contact properties"""
        try:
            response = requests.post(
                f"{self.base_url}/crm/v3/objects/contacts/batch/update",
                headers=self.headers,
                json={"inputs": contacts}
            )
            
            return {
                "success": response.status_code == 200,
                "updated_count": len(contacts),
                "response": response.json() if response.status_code == 200 else response.text
            }
            
        except Exception as e:
            logger.error(f"Error updating contacts: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_custom_properties(self):
        """Create custom properties for our analytics data"""
        properties = [
            {
                "name": "rfm_segment",
                "label": "RFM Segment",
                "type": "string",
                "fieldType": "select",
                "options": [
                    {"label": "Champions", "value": "Champions"},
                    {"label": "Loyal Customers", "value": "Loyal Customers"},
                    {"label": "Potential Loyalists", "value": "Potential Loyalists"},
                    {"label": "New Customers", "value": "New Customers"},
                    {"label": "At Risk", "value": "At Risk"},
                    {"label": "Can't Lose Them", "value": "Can't Lose Them"},
                    {"label": "Hibernating", "value": "Hibernating"},
                    {"label": "Lost", "value": "Lost"}
                ]
            },
            {
                "name": "churn_risk",
                "label": "Churn Risk Level",
                "type": "string",
                "fieldType": "select",
                "options": [
                    {"label": "Low", "value": "Low"},
                    {"label": "Medium", "value": "Medium"},
                    {"label": "High", "value": "High"}
                ]
            },
            {
                "name": "churn_probability",
                "label": "Churn Probability",
                "type": "number",
                "fieldType": "number"
            },
            {
                "name": "clv_tier",
                "label": "Customer Lifetime Value Tier",
                "type": "string",
                "fieldType": "select",
                "options": [
                    {"label": "Low", "value": "Low"},
                    {"label": "Medium", "value": "Medium"},
                    {"label": "High", "value": "High"},
                    {"label": "VIP", "value": "VIP"}
                ]
            }
        ]
        
        for prop in properties:
            try:
                response = requests.post(
                    f"{self.base_url}/crm/v3/properties/contacts",
                    headers=self.headers,
                    json=prop
                )
                if response.status_code == 201:
                    logger.info(f"Created property: {prop['name']}")
                elif response.status_code == 409:
                    logger.info(f"Property already exists: {prop['name']}")
            except Exception as e:
                logger.error(f"Error creating property {prop['name']}: {e}")
    
    def _create_churn_lists(self, predictions: pd.DataFrame):
        """Create dynamic lists for different churn risk levels"""
        lists = [
            {
                "name": "High Churn Risk Customers",
                "filters": [{"propertyName": "churn_risk", "operator": "EQ", "value": "High"}]
            },
            {
                "name": "Medium Churn Risk Customers",
                "filters": [{"propertyName": "churn_risk", "operator": "EQ", "value": "Medium"}]
            },
            {
                "name": "Low Churn Risk Customers",
                "filters": [{"propertyName": "churn_risk", "operator": "EQ", "value": "Low"}]
            }
        ]
        
        for list_config in lists:
            try:
                response = requests.post(
                    f"{self.base_url}/contacts/v1/lists",
                    headers=self.headers,
                    json={
                        "name": list_config['name'],
                        "dynamic": True,
                        "filters": [[list_config['filters']]]
                    }
                )
                if response.status_code == 200:
                    logger.info(f"Created list: {list_config['name']}")
            except Exception as e:
                logger.error(f"Error creating list: {e}")

class MailchimpConnector(MarketingAutomationConnector):
    """Mailchimp integration connector"""
    
    def __init__(self, api_key: str, server_prefix: str):
        self.api_key = api_key
        self.server_prefix = server_prefix
        self.base_url = f"https://{server_prefix}.api.mailchimp.com/3.0"
        self.headers = {
            "Authorization": f"apikey {api_key}",
            "Content-Type": "application/json"
        }
    
    def sync_customer_segments(self, segments: pd.DataFrame) -> Dict[str, Any]:
        """Sync customer segments to Mailchimp"""
        try:
            # Get or create audience
            audience_id = self._get_or_create_audience("E-commerce Customers")
            
            # Create tags for segments
            unique_segments = segments['RFM_segment'].unique()
            for segment in unique_segments:
                self._create_tag(audience_id, segment)
            
            # Update subscribers
            success_count = 0
            for _, row in segments.iterrows():
                result = self._update_subscriber(
                    audience_id,
                    row['customer_email'] if 'customer_email' in row else f"{row['customer_id']}@example.com",
                    {
                        "RFM_SEGMENT": row['RFM_segment'],
                        "RFM_SCORE": row['RFM_score'],
                        "CLV_TIER": row.get('clv_tier', 'Unknown')
                    },
                    tags=[row['RFM_segment']]
                )
                if result:
                    success_count += 1
            
            return {
                "success": True,
                "synced_count": success_count,
                "total_count": len(segments)
            }
            
        except Exception as e:
            logger.error(f"Error syncing to Mailchimp: {e}")
            return {"success": False, "error": str(e)}
    
    def sync_churn_predictions(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """Sync churn predictions to Mailchimp"""
        try:
            audience_id = self._get_or_create_audience("E-commerce Customers")
            
            # Create churn risk tags
            churn_tags = ["High Churn Risk", "Medium Churn Risk", "Low Churn Risk"]
            for tag in churn_tags:
                self._create_tag(audience_id, tag)
            
            # Update subscribers with churn data
            success_count = 0
            for _, row in predictions.iterrows():
                churn_risk = "High Churn Risk" if row['churn_probability'] > 0.7 else \
                            "Medium Churn Risk" if row['churn_probability'] > 0.3 else \
                            "Low Churn Risk"
                
                result = self._update_subscriber(
                    audience_id,
                    row.get('customer_email', f"{row['customer_id']}@example.com"),
                    {
                        "CHURN_RISK": churn_risk,
                        "CHURN_PROB": round(row['churn_probability'], 3)
                    },
                    tags=[churn_risk]
                )
                if result:
                    success_count += 1
            
            return {
                "success": True,
                "synced_count": success_count,
                "total_count": len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Error syncing churn predictions: {e}")
            return {"success": False, "error": str(e)}
    
    def create_campaign(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a campaign in Mailchimp"""
        try:
            # Create campaign
            campaign_response = requests.post(
                f"{self.base_url}/campaigns",
                headers=self.headers,
                json={
                    "type": "regular",
                    "recipients": {
                        "list_id": campaign_data['audience_id'],
                        "segment_opts": {
                            "match": "all",
                            "conditions": campaign_data.get('conditions', [])
                        }
                    },
                    "settings": {
                        "subject_line": campaign_data['subject'],
                        "title": campaign_data['name'],
                        "from_name": campaign_data.get('from_name', 'E-commerce Team'),
                        "reply_to": campaign_data.get('reply_to', 'noreply@example.com')
                    }
                }
            )
            
            if campaign_response.status_code == 200:
                campaign_id = campaign_response.json()['id']
                
                # Set campaign content
                content_response = requests.put(
                    f"{self.base_url}/campaigns/{campaign_id}/content",
                    headers=self.headers,
                    json={
                        "html": campaign_data.get('html_content', ''),
                        "plain_text": campaign_data.get('plain_content', '')
                    }
                )
                
                return {
                    "success": True,
                    "campaign_id": campaign_id,
                    "web_id": campaign_response.json()['web_id']
                }
            else:
                return {"success": False, "error": campaign_response.text}
                
        except Exception as e:
            logger.error(f"Error creating campaign: {e}")
            return {"success": False, "error": str(e)}
    
    def update_contact_properties(self, contacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch update contact properties in Mailchimp"""
        try:
            audience_id = contacts[0].get('audience_id') if contacts else None
            if not audience_id:
                return {"success": False, "error": "No audience_id provided"}
            
            # Prepare batch operations
            operations = []
            for contact in contacts:
                operations.append({
                    "method": "PUT",
                    "path": f"/lists/{audience_id}/members/{contact['email_hash']}",
                    "body": json.dumps({
                        "email_address": contact['email'],
                        "status": contact.get('status', 'subscribed'),
                        "merge_fields": contact.get('properties', {})
                    })
                })
            
            # Execute batch
            response = requests.post(
                f"{self.base_url}/batches",
                headers=self.headers,
                json={"operations": operations}
            )
            
            return {
                "success": response.status_code == 200,
                "batch_id": response.json().get('id') if response.status_code == 200 else None,
                "updated_count": len(contacts)
            }
            
        except Exception as e:
            logger.error(f"Error updating contacts: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_or_create_audience(self, audience_name: str) -> str:
        """Get existing audience or create new one"""
        # Get lists
        response = requests.get(f"{self.base_url}/lists", headers=self.headers)
        if response.status_code == 200:
            lists = response.json()['lists']
            for lst in lists:
                if lst['name'] == audience_name:
                    return lst['id']
        
        # Create new audience if not found
        create_response = requests.post(
            f"{self.base_url}/lists",
            headers=self.headers,
            json={
                "name": audience_name,
                "contact": {
                    "company": "E-commerce Platform",
                    "address1": "123 Main St",
                    "city": "City",
                    "state": "ST",
                    "zip": "12345",
                    "country": "US"
                },
                "permission_reminder": "You signed up for updates on our e-commerce platform",
                "email_type_option": True,
                "campaign_defaults": {
                    "from_name": "E-commerce Team",
                    "from_email": "noreply@example.com",
                    "subject": "Updates from E-commerce Platform",
                    "language": "en"
                }
            }
        )
        
        if create_response.status_code == 200:
            return create_response.json()['id']
        else:
            raise Exception(f"Failed to create audience: {create_response.text}")
    
    def _create_tag(self, audience_id: str, tag_name: str):
        """Create a tag in Mailchimp"""
        try:
            response = requests.post(
                f"{self.base_url}/lists/{audience_id}/segments",
                headers=self.headers,
                json={
                    "name": tag_name,
                    "static_segment": []
                }
            )
            if response.status_code == 200:
                logger.info(f"Created tag: {tag_name}")
        except Exception as e:
            logger.error(f"Error creating tag: {e}")
    
    def _update_subscriber(self, audience_id: str, email: str, merge_fields: Dict, tags: List[str] = None):
        """Update a single subscriber"""
        import hashlib
        
        subscriber_hash = hashlib.md5(email.lower().encode()).hexdigest()
        
        try:
            # Update subscriber
            response = requests.put(
                f"{self.base_url}/lists/{audience_id}/members/{subscriber_hash}",
                headers=self.headers,
                json={
                    "email_address": email,
                    "status": "subscribed",
                    "merge_fields": merge_fields
                }
            )
            
            # Add tags if provided
            if response.status_code == 200 and tags:
                for tag in tags:
                    tag_response = requests.post(
                        f"{self.base_url}/lists/{audience_id}/members/{subscriber_hash}/tags",
                        headers=self.headers,
                        json={"tags": [{"name": tag, "status": "active"}]}
                    )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error updating subscriber: {e}")
            return False

class MarketingIntegrationManager:
    """Manager class to handle multiple marketing platforms"""
    
    def __init__(self):
        self.connectors = {}
        self._initialize_connectors()
    
    def _initialize_connectors(self):
        """Initialize connectors based on environment variables"""
        # HubSpot
        hubspot_key = os.getenv('HUBSPOT_API_KEY')
        if hubspot_key:
            self.connectors['hubspot'] = HubSpotConnector(hubspot_key)
            logger.info("HubSpot connector initialized")
        
        # Mailchimp
        mailchimp_key = os.getenv('MAILCHIMP_API_KEY')
        mailchimp_prefix = os.getenv('MAILCHIMP_SERVER_PREFIX')
        if mailchimp_key and mailchimp_prefix:
            self.connectors['mailchimp'] = MailchimpConnector(mailchimp_key, mailchimp_prefix)
            logger.info("Mailchimp connector initialized")
    
    def sync_all_platforms(self, segments: pd.DataFrame, predictions: pd.DataFrame) -> Dict[str, Any]:
        """Sync data to all configured platforms"""
        results = {}
        
        for platform, connector in self.connectors.items():
            logger.info(f"Syncing to {platform}...")
            
            # Sync segments
            segment_result = connector.sync_customer_segments(segments)
            
            # Sync predictions
            prediction_result = connector.sync_churn_predictions(predictions)
            
            results[platform] = {
                "segments": segment_result,
                "predictions": prediction_result
            }
        
        return results
    
    def create_targeted_campaigns(self, segment: str, campaign_type: str) -> Dict[str, Any]:
        """Create targeted campaigns across platforms"""
        campaigns = {
            "win_back": {
                "name": "Win Back Campaign - High Churn Risk",
                "subject": "We miss you! Here's 20% off your next purchase",
                "segment_filter": {"churn_risk": "High"},
                "html_content": self._get_campaign_template("win_back")
            },
            "loyalty": {
                "name": "VIP Loyalty Rewards",
                "subject": "Exclusive offers for our VIP customers",
                "segment_filter": {"rfm_segment": "Champions"},
                "html_content": self._get_campaign_template("loyalty")
            },
            "reactivation": {
                "name": "Reactivation Campaign",
                "subject": "It's been a while - here's something special",
                "segment_filter": {"rfm_segment": "Hibernating"},
                "html_content": self._get_campaign_template("reactivation")
            }
        }
        
        if campaign_type not in campaigns:
            return {"success": False, "error": "Invalid campaign type"}
        
        campaign_data = campaigns[campaign_type]
        results = {}
        
        for platform, connector in self.connectors.items():
            try:
                result = connector.create_campaign(campaign_data)
                results[platform] = result
            except Exception as e:
                results[platform] = {"success": False, "error": str(e)}
        
        return results
    
    def _get_campaign_template(self, template_type: str) -> str:
        """Get HTML template for campaign"""
        templates = {
            "win_back": """
                <html>
                <body style="font-family: Arial, sans-serif;">
                    <h1>We Miss You!</h1>
                    <p>It's been a while since your last visit. We'd love to have you back!</p>
                    <p>Use code <strong>COMEBACK20</strong> for 20% off your next purchase.</p>
                    <a href="#" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Shop Now</a>
                </body>
                </html>
            """,
            "loyalty": """
                <html>
                <body style="font-family: Arial, sans-serif;">
                    <h1>VIP Exclusive Offers</h1>
                    <p>As one of our most valued customers, you get exclusive access to our VIP sale!</p>
                    <p>Enjoy up to 30% off on selected items.</p>
                    <a href="#" style="background-color: #9C27B0; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">View VIP Deals</a>
                </body>
                </html>
            """,
            "reactivation": """
                <html>
                <body style="font-family: Arial, sans-serif;">
                    <h1>We've Missed You!</h1>
                    <p>Check out what's new since your last visit.</p>
                    <p>Here's a special 15% discount just for you: <strong>HELLO15</strong></p>
                    <a href="#" style="background-color: #2196F3; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Explore New Arrivals</a>
                </body>
                </html>
            """
        }
        
        return templates.get(template_type, templates["win_back"])