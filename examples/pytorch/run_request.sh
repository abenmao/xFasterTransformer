#export no_proxy=localhost
curl http://localhost:8001/v1/completions -H "Content-Type: application/json" -d @data.json | jq '.choices[0].text'
