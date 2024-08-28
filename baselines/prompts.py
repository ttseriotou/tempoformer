llama2_postlevel = {}
llama2_postlevel[5] = {}
llama2_postlevel[10] = {}

mistral2_postlevel = {}
mistral2_postlevel[5] = {}
mistral2_postlevel[10] = {}

llama2_timelinelevel={}
llama2_timelinelevel[5]={}
llama2_timelinelevel[10]={}

mistral2_timelinelevel = {}
mistral2_timelinelevel[5] ={}
mistral2_timelinelevel[10] ={}

llama2_postlevel[5]['lrs'] = f"""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for labeling online Twitter conversations between users.
<</SYS>>

Given the online post of a user in a conversation stream around a rumourous claim on a newsworthy event which it is discussed by tweets in the stream, determine if in the current post there is a switch with respect to the overall stance.
Answer with "none" for either the absence of a switch or cases where the numbers of supporting and opposing posts are equal and with "switch" for switch between the total number of oppositions (querying or denying) and supports or vice versa.

Example 1
Input: @SkyNews And Well Done #youtube for taking it down.
Output: none

Example 2
Input: @PascalSyn @airlivenet but isn't declaring an emergency the proper procedure if you still have control of the situation ?
Output: switch

Example 3
Input: @Mumbobee this is what ppl are not understanding. Muslims get hurt when there prophet gets made a joke out. And some retaliate wrongly.
Output: none

Example 4
Input: .@jashsf @JesseLaGreca I've been in #Ferguson Market and it looks exactly like where the alleged robber was standing.
Output: none

Example 5
Input: @mythos_man @CBCAlerts @CBCNews @JustinTrudeau you are an imbecile and because of imbeciles like you, our lives are in danger
Output: switch

Only return "none" or "switch". 
Limit the answer to 1 word.
[/INST]
"""


llama2_postlevel[5]['topicshift'] = f"""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for labeling human open-domain conversation turns.
<</SYS>>

Given the conversation turn of a human in the conversation, determine if the topic of the conversation is around the original major converation topic or if the conversation has shifted.
Answer with "major" when the conversation turn belongs to the topic with which the conversation commenced with and which is largely talked about and with "shift" when the conversation turn is part of a sub-topic which is a natural digression from the major topic or if it is part of a complete digression from the major topic.

Example 1
Input: and the other half is just standard sized ceilings.
Output: major

Example 2
Input: So, and, and then when you get, you know, when you get into the full-time basis day care,.
Ouptu: shift

Example 3
Input: Well, does it cost money like to use the, to exercise in the weight room or to, uh, to to go swimming.
Output: shift

Example 4
Input: were just my kids' friends,
Output: shift

Example 5
Input: Wonder if it was by one of those famous writers, you know
Output: major

Only return "major" or "shift". 
Limit the answer to 1 word.
[/INST]
</s>
"""

mistral2_postlevel[5]['lrs'] = f"""You are a helpful, respectful and honest assistant for labeling online Twitter conversations between users.
Given the online post of a user in a conversation stream around a rumourous claim on a newsworthy event which it is discussed by tweets in the stream, determine if in the current post there is a switch with respect to the overall stance.
Answer with "none" for either the absence of a switch or cases where the numbers of supporting and opposing posts are equal and with "switch" for switch between the total number of oppositions (querying or denying) and supports or vice versa.
Your task is to assess and categorize post input after <<<>>> into one of the following predefined outputs:

none
switch

You will only respond with the output. Do not include the word "Output". Do not provide explanations or notes.

####
Here are some examples:

Input: @SkyNews And Well Done #youtube for taking it down.
Output: none
Input: @PascalSyn @airlivenet but isn't declaring an emergency the proper procedure if you still have control of the situation ?
Output: switch
Input: @Mumbobee this is what ppl are not understanding. Muslims get hurt when there prophet gets made a joke out. And some retaliate wrongly.
Output: none
Input: .@jashsf @JesseLaGreca I've been in #Ferguson Market and it looks exactly like where the alleged robber was standing.
Output: none
Input: @mythos_man @CBCAlerts @CBCNews @JustinTrudeau you are an imbecile and because of imbeciles like you, our lives are in danger
Output: switch
###

"""


mistral2_postlevel[5]['topicshift'] = f"""You are a helpful, respectful and honest assistant for labeling human open-domain conversation turns.
Given the conversation turn of a human in the conversation, determine if the topic of the conversation is around the original major converation topic or if the conversation topic has shifted.
Answer with "major" when the conversation turn belongs to the topic with which the conversation commenced with and which is largely talked about and with "shift" when the conversation turn is part of a sub-topic which is a natural digression from the major topic or if it is part of a complete digression from the major topic.
Your task is to assess and categorize user input after <<<>>> into one of the following predefined outputs:

major
shift

You will only respond with the output. Do not include the word "Output". Do not provide explanations or notes.

####
Here are some examples:

Input: and the other half is just standard sized ceilings.
Output: major
Input: So, and, and then when you get, you know, when you get into the full-time basis day care,.
Output: shift
Input: Well, does it cost money like to use the, to exercise in the weight room or to, uh, to to go swimming.
Output: shift
Input: were just my kids' friends,
Output: shift
Input: Wonder if it was by one of those famous writers, you know
Output: major
###

"""

mistral2_postlevel[10]['lrs'] = f"""You are a helpful, respectful and honest assistant for labeling online Twitter conversations between users.
Given the online post of a user in a conversation stream around a rumourous claim on a newsworthy event which it is discussed by tweets in the stream, determine if in the current post there is a switch with respect to the overall stance.
Answer with "none" for either the absence of a switch or cases where the numbers of supporting and opposing posts are equal and with "switch" for switch between the total number of oppositions (querying or denying) and supports or vice versa.
Your task is to assess and categorize post input after <<<>>> into one of the following predefined outputs:

none
switch

You will only respond with the output. Do not include the word "Output". Do not provide explanations or notes.

####
Here are some examples:

Input: @SkyNews And Well Done #youtube for taking it down.
Output: none
Input: @PascalSyn @airlivenet but isn't declaring an emergency the proper procedure if you still have control of the situation ?
Output: switch
Input: @Mumbobee this is what ppl are not understanding. Muslims get hurt when there prophet gets made a joke out. And some retaliate wrongly.
Output: none
Input: .@jashsf @JesseLaGreca I've been in #Ferguson Market and it looks exactly like where the alleged robber was standing.
Output: none
Input: @mythos_man @CBCAlerts @CBCNews @JustinTrudeau you are an imbecile and because of imbeciles like you, our lives are in danger
Output: switch
Input: @ivoirien_com @CBCAlerts I know I can't be insightful yet - that's the bloody point, neither can you, and by trying to, you lose ground
Output: switch
Input: @ericabecks @MissTaraTeng @globeandmail who said anything about isis?
Output: switch
Input: @CarmelP92 @SkyNews just seen it, jeez :/
Output: none
Input: @TroyBramston @australian @channeltennews pathetic islamist coward
Output: none
Input: @WSJ now remember, even if your kids r in the cafe, no torturing anyone who might know anything if this happens in U.S.
Output: none
###

"""

mistral2_postlevel[10]['topicshift'] = f"""You are a helpful, respectful and honest assistant for labeling human open-domain conversation turns.
Given the conversation turn of a human in the conversation, determine if the topic of the conversation is around the original major converation topic or if the conversation topic has shifted.
Answer with "major" when the conversation turn belongs to the topic with which the conversation commenced with and which is largely talked about and with "shift" when the conversation turn is part of a sub-topic which is a natural digression from the major topic or if it is part of a complete digression from the major topic.
Your task is to assess and categorize user input after <<<>>> into one of the following predefined outputs:

major
shift

You will only respond with the output. Do not include the word "Output". Do not provide explanations or notes.

####
Here are some examples:

Input: and the other half is just standard sized ceilings.
Output: major
Input: So, and, and then when you get, you know, when you get into the full-time basis day care,.
Output: shift
Input: Well, does it cost money like to use the, to exercise in the weight room or to, uh, to to go swimming.
Output: shift
Input: were just my kids' friends,
Output: shift
Input: Wonder if it was by one of those famous writers, you know
Output: major
Input: Every month I add two hundred and twenty dollars to it.
Output: shift
Input: It's just there's, there's no motivation.
Output: shift
Input: But, or, or even somewhere better,
Output: shift
Input: there you go.
Output: major
Input: and, um, though he's alive today and everything worked out fine, it, it, it happened about two blocks from the high school
Output: major
###

"""


llama2_postlevel[10]['lrs'] = f"""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for labeling online Twitter conversations between users.
<</SYS>>

Given the online post of a user in a conversation stream around a rumourous claim on a newsworthy event which it is discussed by tweets in the stream, determine if in the current post there is a switch with respect to the overall stance.
Answer with "none" for either the absence of a switch or cases where the numbers of supporting and opposing posts are equal and with "switch" for switch between the total number of oppositions (querying or denying) and supports or vice versa.

Example 1
Input: @SkyNews And Well Done #youtube for taking it down.
Output: none

Example 2
Input: @PascalSyn @airlivenet but isn't declaring an emergency the proper procedure if you still have control of the situation ?
Output: switch

Example 3
Input: @Mumbobee this is what ppl are not understanding. Muslims get hurt when there prophet gets made a joke out. And some retaliate wrongly.
Output: none

Example 4
Input: .@jashsf @JesseLaGreca I've been in #Ferguson Market and it looks exactly like where the alleged robber was standing.
Output: none

Example 5
Input: @mythos_man @CBCAlerts @CBCNews @JustinTrudeau you are an imbecile and because of imbeciles like you, our lives are in danger
Output: switch

Example 6
Input: @ivoirien_com @CBCAlerts I know I can't be insightful yet - that's the bloody point, neither can you, and by trying to, you lose ground
Output: switch

Example 7
Input: @ericabecks @MissTaraTeng @globeandmail who said anything about isis?
Output: switch

Example 8
Input: @CarmelP92 @SkyNews just seen it, jeez :/
Output: none

Example 9
Input: @TroyBramston @australian @channeltennews pathetic islamist coward
Output: none

Example 10
Input: @WSJ now remember, even if your kids r in the cafe, no torturing anyone who might know anything if this happens in U.S.
Output: none

Only return "none" or "switch". 
Limit the answer to 1 word.
[/INST]
"""

llama2_postlevel[10]['topicshift'] = f"""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for labeling human open-domain conversation turns.
<</SYS>>

Given the conversation turn of a human in the conversation, determine if the topic of the conversation is around the original major converation topic or if the conversation has shifted.
Answer with "major" when the conversation turn belongs to the topic with which the conversation commenced with and which is largely talked about and with "shift" when the conversation turn is part of a sub-topic which is a natural digression from the major topic or if it is part of a complete digression from the major topic.

Example 1
Input: and the other half is just standard sized ceilings.
Output: major

Example 2
Input: So, and, and then when you get, you know, when you get into the full-time basis day care,.
Ouptu: shift

Example 3
Input: Well, does it cost money like to use the, to exercise in the weight room or to, uh, to to go swimming.
Output: shift

Example 4
Input: were just my kids' friends,
Output: shift

Example 5
Input: Wonder if it was by one of those famous writers, you know
Output: major

Example 6
Input: Every month I add two hundred and twenty dollars to it.
Output: shift

Example 7
Input: It's just there's, there's no motivation.
Output: shift

Example 8
Input: But, or, or even somewhere better,
Output: shift

Example 9
Input: there you go.
Output: major

Example 10
Input: and, um, though he's alive today and everything worked out fine, it, it, it happened about two blocks from the high school
Output: major

Only return "major" or "shift". 
Limit the answer to 1 word.
[/INST]
</s>
"""


llama2_timelinelevel[5]['lrs'] = f"""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for labeling online Twitter conversations between users.
<</SYS>>

Given the most recent online conversation history between users around a rumourous claim on a newsworthy event, determine if the most recent input user post is a switch with respect to the overall conversation stance.
Answer with "none" for either the absence of a switch or cases where the numbers of supporting and opposing posts are equal and with "switch" for switch between the total number of oppositions (querying or denying) and supports or vice versa.

Example 1
Conversation History:
TV channels have chosen not to show videos of hostages relaying #sydneysiege gunman's demands. http://t.co/5jtd36wFOU http://t.co/oKJPRv40l3
@SkyNews Its the right thing to do
@SkyNews the internet says lol
@SkyNews Just shoot him! Marksman have clearly had enough opportunities to do so
Input: @SkyNews And Well Done #youtube for taking it down.
Output: none

Example 2
Conversation History:
I doubt any pilot would say \'Emergency\', but rather \'Mayday\' RT "@airlivenet NEWS: \'Emergency Emergency\' was the final distress call #4U9525
@PascalSyn @airlivenet it's a scale. Emergency &lt; Mayday
@damienramage @airlivenet No, then you would say PANPAN. Trust me, I'm a pilot! Besides: Mayday is when lives are in danger...
@airlivenet @PollyR_Aviation @crislomb Looks terribly off course on a map. Nothing earlier? Like much earlier?
Input: @PascalSyn @airlivenet but isn't declaring an emergency the proper procedure if you still have control of the situation ?
Output: switch

Example 3
Conversation History:
@UnbiasedF If you go into such facts it will be a very long story to keep it short that 1.6 bil ppl can't have a prophet made a joke out.
@Cavelloman1 @mjhsinclair @tnewtondunn @VictoriaPeckham and give true meaning to the hashtag #illridewithyou #charliehebdo.
@Olenkafrenkiel @Cavelloman1 @mjhsinclair @tnewtondunn @VictoriaPeckham I'll ride with you. #illridewithyou http://t.co/dFLDTDqTbS
@m33ryg @katherine1924 @tnewtondunn @mehdirhasan This is the point of free speech, why get so offended!!?
Input: @Mumbobee this is what ppl are not understanding. Muslims get hurt when there prophet gets made a joke out. And some retaliate wrongly.
Output: none

Example 4
Conversation History:
@JustinGlawe @YourAnonCentral SIGN AND SHARE https://t.co/NRV4SXSuSL #Ferguson #JusticeForMikeBrown
@JustinGlawe Who are you to expect any type of answer in the first place?
@marymolina20 a person who makes his living asking questions, that's who.
@JustinGlawe @JesseLaGreca it wasn't the quik stop that got burned down?
Input: .@jashsf @JesseLaGreca I've been in #Ferguson Market and it looks exactly like where the alleged robber was standing.
Output: none

Example 5
Conversation History:
@mythos_man @CBCAlerts @CBCNews @JustinTrudeau your #Harper govt has changed Canada to a military nation...
@mythos_man @CBCAlerts @CBCNews @JustinTrudeau when our sole purpose was to help civilians with humanitarian aid in the past under liberals
@DaryusAzad @CBCAlerts @CBCNews @JustinTrudeau you force that woman in your pic to dress that way?
@Cdn_Bowhunter @CBCAlerts @CBCNews @JustinTrudeau exactly so why did you or your friend already pointed the fingers at Muslims #ignorantmuch
Input: @mythos_man @CBCAlerts @CBCNews @JustinTrudeau you are an imbecile and because of imbeciles like you, our lives are in danger
Output: switch

Only return "none" or "switch". 
Limit the answer to 1 word.
[/INST]
</s>

"""


llama2_timelinelevel[5]['topicshift'] = f"""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for labeling human open-domain conversation turns.
<</SYS>>

Given the most recent conversation turn history between humans in the conversation, determine if at the most recent input conversation turn the topic of the conversation is around the original major converation topic or if the conversation topic has shifted.
Answer with "major" when the conversation turn belongs to the topic with which the conversation commenced with and which is largely talked about and with "shift" when the conversation turn is part of a sub-topic which is a natural digression from the major topic or if it is part of a complete digression from the major topic.

Example 1
Conversation History:
Uh, and it's, uh, I guess what they call a story and a half.
Because it's not a full two story. Where, you know, everything on top is on bottom.
Yeah.
So, it's got real high ceilings on half the house
Input: and the other half is just standard sized ceilings.
Output: major

Example 2
Conversation History:
but that's always a little bit more expensive than what I could look at.
Uh-huh.
Um, and I was very very fortunate in that I didn't have to do that on a full-time basis.
Yes.
Input: So, and, and then when you get, you know, when you get into the full-time basis day care,.
Output: shift

Example 3
Conversation History:
I just don't take as much advantage of it as I should.
Now to use,
I mean, I'm just now finally starting to do the aerobics thing.
But.
Input: Well, does it cost money like to use the, to exercise in the weight room or to, uh, to to go swimming.
Output: shift

Example 4
Conversation History:
and, uh, they both got ran over. After we had them for several years, just nice little outside dogs
Oh, no.
but
Yeah.
Input: were just my kids' friends,
Output: shift

Example 5
Conversation History:
it was some time ago.
Seems like it was THE SPY WENT DANCING, or something like that.
Oh, it sounds like fun.
It was a lot of fun because they used these real names, you know.
Input: Wonder if it was by one of those famous writers, you know
Output: major

Only return "major" or "shift". 
Limit the answer to 1 word.
[/INST]
</s>

"""



mistral2_timelinelevel[5]['lrs'] = f"""You are a helpful, respectful and honest assistant for labeling online Twitter conversations between users.
Given the most recent online conversation history between users around a rumourous claim on a newsworthy event, determine if the most recent input user post is a switch with respect to the overall conversation stance.
Answer with "none" for either the absence of a switch or cases where the numbers of supporting and opposing posts are equal and with "switch" for switch between the total number of oppositions (querying or denying) and supports or vice versa.
Your task is to assess and categorize post input after <<<>>> into one of the following predefined outputs:

none
switch

You will only respond with the output. Do not include the word "Output". Do not provide explanations or notes.

####
Here are some examples:

Conversation History:
TV channels have chosen not to show videos of hostages relaying #sydneysiege gunman's demands. http://t.co/5jtd36wFOU http://t.co/oKJPRv40l3
@SkyNews Its the right thing to do
@SkyNews the internet says lol
@SkyNews Just shoot him! Marksman have clearly had enough opportunities to do so
Input: @SkyNews And Well Done #youtube for taking it down.
Output: none

Conversation History:
I doubt any pilot would say \'Emergency\', but rather \'Mayday\' RT "@airlivenet NEWS: \'Emergency Emergency\' was the final distress call #4U9525
@PascalSyn @airlivenet it's a scale. Emergency &lt; Mayday
@damienramage @airlivenet No, then you would say PANPAN. Trust me, I'm a pilot! Besides: Mayday is when lives are in danger...
@airlivenet @PollyR_Aviation @crislomb Looks terribly off course on a map. Nothing earlier? Like much earlier?
Input: @PascalSyn @airlivenet but isn't declaring an emergency the proper procedure if you still have control of the situation ?
Output: switch

Conversation History:
@UnbiasedF If you go into such facts it will be a very long story to keep it short that 1.6 bil ppl can't have a prophet made a joke out.
@Cavelloman1 @mjhsinclair @tnewtondunn @VictoriaPeckham and give true meaning to the hashtag #illridewithyou #charliehebdo.
@Olenkafrenkiel @Cavelloman1 @mjhsinclair @tnewtondunn @VictoriaPeckham I'll ride with you. #illridewithyou http://t.co/dFLDTDqTbS
@m33ryg @katherine1924 @tnewtondunn @mehdirhasan This is the point of free speech, why get so offended!!?
Input: @Mumbobee this is what ppl are not understanding. Muslims get hurt when there prophet gets made a joke out. And some retaliate wrongly.
Output: none

Conversation History:
@JustinGlawe @YourAnonCentral SIGN AND SHARE https://t.co/NRV4SXSuSL #Ferguson #JusticeForMikeBrown
@JustinGlawe Who are you to expect any type of answer in the first place?
@marymolina20 a person who makes his living asking questions, that's who.
@JustinGlawe @JesseLaGreca it wasn't the quik stop that got burned down?
Input: .@jashsf @JesseLaGreca I've been in #Ferguson Market and it looks exactly like where the alleged robber was standing.
Output: none

Conversation History:
@mythos_man @CBCAlerts @CBCNews @JustinTrudeau your #Harper govt has changed Canada to a military nation...
@mythos_man @CBCAlerts @CBCNews @JustinTrudeau when our sole purpose was to help civilians with humanitarian aid in the past under liberals
@DaryusAzad @CBCAlerts @CBCNews @JustinTrudeau you force that woman in your pic to dress that way?
@Cdn_Bowhunter @CBCAlerts @CBCNews @JustinTrudeau exactly so why did you or your friend already pointed the fingers at Muslims #ignorantmuch
Input: @mythos_man @CBCAlerts @CBCNews @JustinTrudeau you are an imbecile and because of imbeciles like you, our lives are in danger
Output: switch
###

"""


mistral2_timelinelevel[5]['topicshift'] = f"""You are a helpful, respectful and honest assistant for labeling human open-domain conversation turns.
Given the most recent conversation turn history between humans in the conversation, determine if at the most recent input conversation turn the topic of the conversation is around the original major converation topic or if the conversation topic has shifted.
Answer with "major" when the conversation turn belongs to the topic with which the conversation commenced with and which is largely talked about and with "shift" when the conversation turn is part of a sub-topic which is a natural digression from the major topic or if it is part of a complete digression from the major topic.
Your task is to assess and categorize user input after <<<>>> into one of the following predefined outputs:

major
shift

You will only respond with the output. Do not include the word "Output". Do not provide explanations or notes.

####
Here are some examples:

Conversation History:
Uh, and it's, uh, I guess what they call a story and a half.
Because it's not a full two story. Where, you know, everything on top is on bottom.
Yeah.
So, it's got real high ceilings on half the house
Input: and the other half is just standard sized ceilings.
Output: major

Conversation History:
but that's always a little bit more expensive than what I could look at.
Uh-huh.
Um, and I was very very fortunate in that I didn't have to do that on a full-time basis.
Yes.
Input: So, and, and then when you get, you know, when you get into the full-time basis day care,.
Output: shift

Conversation History:
I just don't take as much advantage of it as I should.
Now to use,
I mean, I'm just now finally starting to do the aerobics thing.
But.
Input: Well, does it cost money like to use the, to exercise in the weight room or to, uh, to to go swimming.
Output: shift

Conversation History:
and, uh, they both got ran over. After we had them for several years, just nice little outside dogs
Oh, no.
but
Yeah.
Input: were just my kids' friends,
Output: shift

Conversation History:
it was some time ago.
Seems like it was THE SPY WENT DANCING, or something like that.
Oh, it sounds like fun.
It was a lot of fun because they used these real names, you know.
Input: Wonder if it was by one of those famous writers, you know
Output: major
###

"""


mistral2_timelinelevel[10]['lrs'] = f"""You are a helpful, respectful and honest assistant for labeling online Twitter conversations between users.
Given the most recent online conversation history between users around a rumourous claim on a newsworthy event, determine if the most recent input user post is a switch with respect to the overall conversation stance.
Answer with "none" for either the absence of a switch or cases where the numbers of supporting and opposing posts are equal and with "switch" for switch between the total number of oppositions (querying or denying) and supports or vice versa.
Your task is to assess and categorize post input after <<<>>> into one of the following predefined outputs:

none
switch

You will only respond with the output. Do not include the word "Output". Do not provide explanations or notes.

####
Here are some examples:

Conversation History:
TV channels have chosen not to show videos of hostages relaying #sydneysiege gunman's demands. http://t.co/5jtd36wFOU http://t.co/oKJPRv40l3
@SkyNews Its the right thing to do
@SkyNews the internet says lol
@SkyNews Just shoot him! Marksman have clearly had enough opportunities to do so
Input: @SkyNews And Well Done #youtube for taking it down.
Output: none

Conversation History:
I doubt any pilot would say \'Emergency\', but rather \'Mayday\' RT "@airlivenet NEWS: \'Emergency Emergency\' was the final distress call #4U9525
@PascalSyn @airlivenet it's a scale. Emergency &lt; Mayday
@damienramage @airlivenet No, then you would say PANPAN. Trust me, I'm a pilot! Besides: Mayday is when lives are in danger...
@airlivenet @PollyR_Aviation @crislomb Looks terribly off course on a map. Nothing earlier? Like much earlier?
Input: @PascalSyn @airlivenet but isn't declaring an emergency the proper procedure if you still have control of the situation ?
Output: switch

Conversation History:
@UnbiasedF If you go into such facts it will be a very long story to keep it short that 1.6 bil ppl can't have a prophet made a joke out.
@Cavelloman1 @mjhsinclair @tnewtondunn @VictoriaPeckham and give true meaning to the hashtag #illridewithyou #charliehebdo.
@Olenkafrenkiel @Cavelloman1 @mjhsinclair @tnewtondunn @VictoriaPeckham I'll ride with you. #illridewithyou http://t.co/dFLDTDqTbS
@m33ryg @katherine1924 @tnewtondunn @mehdirhasan This is the point of free speech, why get so offended!!?
Input: @Mumbobee this is what ppl are not understanding. Muslims get hurt when there prophet gets made a joke out. And some retaliate wrongly.
Output: none

Conversation History:
@JustinGlawe @YourAnonCentral SIGN AND SHARE https://t.co/NRV4SXSuSL #Ferguson #JusticeForMikeBrown
@JustinGlawe Who are you to expect any type of answer in the first place?
@marymolina20 a person who makes his living asking questions, that's who.
@JustinGlawe @JesseLaGreca it wasn't the quik stop that got burned down?
Input: .@jashsf @JesseLaGreca I've been in #Ferguson Market and it looks exactly like where the alleged robber was standing.
Output: none

Conversation History:
@mythos_man @CBCAlerts @CBCNews @JustinTrudeau your #Harper govt has changed Canada to a military nation...
@mythos_man @CBCAlerts @CBCNews @JustinTrudeau when our sole purpose was to help civilians with humanitarian aid in the past under liberals
@DaryusAzad @CBCAlerts @CBCNews @JustinTrudeau you force that woman in your pic to dress that way?
@Cdn_Bowhunter @CBCAlerts @CBCNews @JustinTrudeau exactly so why did you or your friend already pointed the fingers at Muslims #ignorantmuch
Input: @mythos_man @CBCAlerts @CBCNews @JustinTrudeau you are an imbecile and because of imbeciles like you, our lives are in danger
Output: switch

Conversation History:
@CBCAlerts We are not obligated to sit passively, just waiting for these rapists to tear up our society!  Arrest them NOW!!!
@CBCAlerts Change the laws, change the reckless immigration policies!  Sensitize people to be on the look-out!  Get real, folks!
@ivoirien_com @CBCAlerts Can you lay off the politics right now? We know *nothing* about who did this, what's going on.
@ashleyjmorton @CBCAlerts Right, there is no obvious context.  How insightful of you!
Input: @ivoirien_com @CBCAlerts I know I can't be insightful yet - that's the bloody point, neither can you, and by trying to, you lose ground
Output: switch

Conversation History:
@IStateYourName_ @globeandmail @exjon Are you on strong opiates or are you an idiot?
@MissTaraTeng @globeandmail I say he was let in that is the only explanation. We have to b open If it is ISIS related they have supporters
@MissTaraTeng @globeandmail that 1st gunman is dead his partner in crime is on a brown motorcycle and took shelter
@cfletch_13 @globeandmail @exjon Neither. Do you not understand how sarcasm works or are you merely obtuse?
Input: @ericabecks @MissTaraTeng @globeandmail who said anything about isis?
Output: switch

Conversation History:
"@SkyNews: Live Updates: Siege in Sydney Cafe http://t.co/dhzIV6TVi7"       @emfletch92
@SkyNews Seriously fed up with this. It has to be stopped. Nowhere is safe nowadays. It's Sydney! Australia doesn't need this s**t now too!
@SkyNews bollox!
"@SkyNews: Live Updates: Siege in Sydney Cafe http://t.co/YiKuQUAAWI"
Input: @CarmelP92 @SkyNews just seen it, jeez :/
Output: none

Conversation History:
@TroyBramston @australian @channeltennews is this same as the "call" to Ray Hadley by "POI" from 10.30am? Or new info?
@TroyBramston @channeltennews fool bought the wrong flag with him.
@TroyBramston @channeltennews Deliver him the Australian SAS instead.
@TroyBramston @srod009 @channeltennews haha getting your news from 10 is like watching Looney Tunes for Social skills ha #MartinPlace
Input: @TroyBramston @australian @channeltennews pathetic islamist coward
Output: none

Conversation History:
@WSJ @maybe Sydney needs to borrow some guns??
@WSJ More workplace violence I'm sure !
"@WSJ: Central Sydney shut down by police amid ongoing hostage situation: http://t.co/Cla9yAVq4P" hoping for all the best here
@WSJ @dwinningWSJ \nPrayers
Input: @WSJ now remember, even if your kids r in the cafe, no torturing anyone who might know anything if this happens in U.S.
Output: none
###

"""

mistral2_timelinelevel[10]['topicshift'] = f"""You are a helpful, respectful and honest assistant for labeling human open-domain conversation turns.
Given the most recent conversation turn history between humans in the conversation, determine if at the most recent input conversation turn the topic of the conversation is around the original major converation topic or if the conversation topic has shifted.
Answer with "major" when the conversation turn belongs to the topic with which the conversation commenced with and which is largely talked about and with "shift" when the conversation turn is part of a sub-topic which is a natural digression from the major topic or if it is part of a complete digression from the major topic.
Your task is to assess and categorize user input after <<<>>> into one of the following predefined outputs:

major
shift

You will only respond with the output. Do not include the word "Output". Do not provide explanations or notes.

####
Here are some examples:

Conversation History:
Uh, and it's, uh, I guess what they call a story and a half.
Because it's not a full two story. Where, you know, everything on top is on bottom.
Yeah.
So, it's got real high ceilings on half the house
Input: and the other half is just standard sized ceilings.
Output: major

Conversation History:
but that's always a little bit more expensive than what I could look at.
Uh-huh.
Um, and I was very very fortunate in that I didn't have to do that on a full-time basis.
Yes.
Input: So, and, and then when you get, you know, when you get into the full-time basis day care,.
Output: shift

Conversation History:
I just don't take as much advantage of it as I should.
Now to use,
I mean, I'm just now finally starting to do the aerobics thing.
But.
Input: Well, does it cost money like to use the, to exercise in the weight room or to, uh, to to go swimming.
Output: shift

Conversation History:
and, uh, they both got ran over. After we had them for several years, just nice little outside dogs
Oh, no.
but
Yeah.
Input: were just my kids' friends,
Output: shift

Conversation History:
it was some time ago.
Seems like it was THE SPY WENT DANCING, or something like that.
Oh, it sounds like fun.
It was a lot of fun because they used these real names, you know.
Input: Wonder if it was by one of those famous writers, you know
Output: major

Conversation History:
but, what I do on my money market account, my taxes for instance which amount to an average of two hundred and twenty dollars a month. I will just take two hundred and twenty out
and I put it in parenthesis.
I take it out of the right line total and put it in parenthesis in there and let it build up.
Uh-huh.
Input: Every month I add two hundred and twenty dollars to it.
Output: shift

Conversation History:
but,
I don't know
being a, being the bachelor here that I am, I, I don't think I'd, I'd be
I mean it, it's not a question of capability here to do something like that.
Input: It's just there's, there's no motivation.
Output: shift

Conversation History:
Well, would you like to be there?
I,
all things being equal, yes.
Uh-huh.
Input: But, or, or even somewhere better,
Output: shift

Conversation History:
And, and I'm looking at it as a change.
Uh-huh.
And I'm looking at it as, as an adventure.
Right,
Input: there you go.
Output: major

Conversation History:
it's really funny though, um,
I guess, my dad was hit by a, a drunk driver who was also high um, when I was probably eighteen, my senior year in high school
Um.
Oh, my goodness.
Input: and, um, though he's alive today and everything worked out fine, it, it, it happened about two blocks from the high school
Output: major
###

"""

