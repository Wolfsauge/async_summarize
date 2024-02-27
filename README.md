<img align="right" src="images/async_summarize.png">

# async_summarize
An asynchronous summarization script.

This script summarizes the input file using a large language model API. If the input exceeds the context of the LLM, it will be split using [semantic-text-splitter](https://pypi.org/project/semantic-text-splitter/).

# Note

The documentation needs rework ...

## Example call

```
$ ./main.py -u http://localhost:8000/v1 \
    -p prompt-mistral_instruct_v0.2.yaml \
    -f mobydick.txt
```

## Example result

> In Herman Melville's "Moby-Dick," Ishmael and Queequeg embark on a whaling voyage, sharing a bond as they face the challenges of the journey. Upon reaching Nantucket, they join the crew of the Pequod, led by Captain Ahab, who is notoriously obsessed with hunting the white whale, Moby-Dick. Ahab offers a gold piece to the first one who harpoons the white whale, but Starbuck, the first mate, expresses reluctance due to Ahab's personal vengeance.
>
> As they set sail, the crew experiences the thrill and danger of whale hunting during the first lowering for a sperm whale. The text emphasizes the camaraderie among the crew and the awe-inspiring nature of the chase. Superstitions and beliefs surrounding the whale's appearance come to the forefront, with the mysterious Fedallah playing a role on the ship. The text also discusses the bond between Ishmael and Queequeg, exploring themes of respect for different beliefs, the importance of staying true to one's course, and the obsession that drives Ahab's quest for revenge.
>
> The text describes various representations of whales throughout history, from ancient sculptures to scientific illustrations, and reflects on the inaccuracies and limitations of these depictions. The author argues that the true form of the whale can only be seen at sea and cannot be accurately captured in a portrait. The text mentions several inaccurate depictions, including those in old Hindu sculpture, European paintings, and scientific books. The skeleton of a whale does not provide an accurate representation of its living form.
>
> The chapter also includes a description of the sighting of a giant squid and the use of whale-lines in whaling. The author emphasizes the danger and complexity of the whale-line, which envelops the entire boat and puts the crew in peril. The text concludes by reflecting on the inherent dangers and mysteries of life and the calm before the storm. The awe-inspiring and terrifying nature of the sea and its creatures is compared to the savagery of a whale hunter and the wonder of a Hawaiian war-club.
