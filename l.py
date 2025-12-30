import boto3
import json

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

def lambda_handler(event, context):
    # Handle preflight (OPTIONS) request for CORS
    if event['httpMethod'] == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps('Preflight OK')
        }

    try:
        # Parse incoming JSON
        body = json.loads(event['body'])
        user_message = body.get('message', '')
        history = body.get('history', [])

        # Optional: construct prompt with history
        conversation = ""
        for turn in history:
            conversation += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        conversation += f"User: {user_message}\nAssistant:"

        # Create payload for Titan
        request_body = {
            "inputText": conversation,
            "textGenerationConfig": {
                "maxTokenCount": 300,
                "temperature": 0.7,
                "topP": 0.9,
                "stopSequences": []
            }
        }

        # Call Titan model
        response = bedrock.invoke_model(
            modelId='amazon.titan-text-express-v1',
            body=json.dumps(request_body),
            contentType='application/json',
            accept='application/json'
        )

        # Extract model output
        result = json.loads(response['body'].read())
        
        # Check if response contains the expected result
        if 'results' in result and len(result['results']) > 0:
            reply = result['results'][0].get('outputText', '')
        else:
            reply = "Sorry, I couldn't generate a response."

        # Return response with CORS headers
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps({'response': reply})
        }

    except Exception as e:
        # Error handling for any issues in the Lambda function
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps({'error': str(e)})
        }
