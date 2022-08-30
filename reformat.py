import json

def main():
  with open("moneymx_reformat.jsonl", 'w') as w:
    with open("moneymx.jsonl", 'r') as r:
      for l in r.readlines():
        event = json.loads(l)
        w.write(json.dumps({
          "the_id": event['event_id'],
          "_type": event['event_type'], 
          "_time": event['event_ts'], 
          **event['event']
        }) + '\n')

if __name__ == "__main__":
  main()
