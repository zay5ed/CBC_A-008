import csv
import openai
from flask import Flask, request, jsonify

# Initialize the Flask app
app = Flask(__name__)

# Set OpenAI API key
openai.api_key = "your-openai-api-key"  # Replace with your OpenAI API key

# Parse the CSV file for interest and link data
def load_data():
    people_data = {}
    with open('people_interests.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            people_data[row['Name']] = {
                'interests': [row[f"Interest {i}"] for i in range(1, 4)],
                'links': {row[f"Interest {i}"]: [row[f"Links {j}"] for j in range(1, 5)] for i in range(1, 4)}
            }
    return people_data

people_data = load_data()

@app.route('/get_flowchart', methods=['POST'])
def get_flowchart():
    data = request.get_json()
    interests = data['interests']

    # Construct a prompt to generate a flowchart response
    flowchart_text = ""
    for interest in interests:
        links = people_data.get("Person 1", {}).get('links', {}).get(interest, [])
        if links:
            flowchart_text += f"\nInterest: {interest}\n"
            for idx, link in enumerate(links, 1):
                flowchart_text += f"  - Link {idx}: {link}\n"
        else:
            flowchart_text += f"\nInterest: {interest} has no links available.\n"

    # Generate response using OpenAI API
    prompt = f"Generate a flowchart for the following interests: {flowchart_text}"
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can use GPT-4 as well
        prompt=prompt,
        max_tokens=200,
        temperature=0.7
    )

    return jsonify({'flowchart': response.choices[0].text.strip()})

if __name__ == '__main__':
    app.run(debug=True)
