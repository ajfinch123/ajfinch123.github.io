---
layout: post
title:  "Is Bigger Really Better?"
date:   2020-08-31 20:00:27 +0000
categories: ML AI Unsupervised Compression
---

# Bigger Is Better

It is axiomatic that bigger is better.

<img src="/assets/img/unsplash_dream_big.jpg" style="width:750px">

I mean, who *doesn't* want a bigger paycheck, right?


But what if - and you may have to sit down for this - what if it *wasn't*?

# Bigger Sucks

<img src="/assets/img/unsplash_big_sucks.jpg" style="width:750px">

Not to get into a social commentary here, but bigger things actually kind of suck.

The reason that we obsess over having big things is that there's a higher likelihood that they'll have (or be able to give us) the small things that we actually need.  

Imagine, for instance, that you needed a needle.  For the sake of argument, let's say that you can't go out and buy one.  Instead, the only way that you're ever going to be able to get a needle is to purchase haystacks and search for one that just *happens* to have a needle in it.  In that case, you're probably going to want **a lot** of haystacks, since each one is individually pretty unlikely to have a needle in it.

(I mean, I'm assuming this to be true.  Full disclosure, I have never actually searched for a needle in a haystack and have radically uninformed prior beliefs about the probability of any haystack containing a needle.)

The thing is, you don't actually need any of this hay.  In fact, the second that you get all of this hay, you're probably going to ask yourself whether it's really worth it to go through all of this searching just to get a needle.  Once you have your needle, you'll probably be content to just throw everything else away - assuming you're confident that nothing else valuable is in there.

Much like our giant piles of hay, big data is only useful for the insights it may contain.  The more data we have, the more likely it is that there's a useful needle lurking in there somewhere.  The question, as always, is how we find the needles.

Let's suppose that we're lazy (a pretty good assumption).  We're going to send out a few little helper robots to make the searching easier.  Unfortunately, these robots wouldn't know a needle if it stabbed them in the little robot paw - literally.  If we had some needles on-hand, we could probably show them to the robots, and they may be able to learn what to look for.  But then again, if we had needles on-hand, we wouldn't have had to get the giant pile of hay in the first place.  Maybe the robots could bring us everything they found, but that would waste a lot of our valuable time...

How do we get our robots to find the needle for us without first having a needle?

# Unsupervised Learning

Enter *Unsupervised Learning.*

We don't have any needles to show our robots and they don't know what to look for.  However, these robots are also pretty smart.  They can learn some things on their own (hence, unsupervised learning).

Here are a few of the things we could try with our robot friends:

* Find all the weird things.

<img src="/assets/img/unsplash_outlier.jpg">

In this strategy, we want our robots to go through every single thing that they find (all of the hay, lost toys, string, rodents, and possibly needles) in every single haystack.  Then, isolate all of the things that don't look like the others.  This is probably a really good strategy in this case, since our robots would have **a lot** of hay and not much else.  At the end, our robots might take us a pile of weird things that didn't look like hay, and we could sort through this much smaller pile to find our needles.

* Take us one copy of every type of item.

<img src= "/assets/img/unsplash_fruits.jpg" style="height:500px">

This is a similar tactic to the last one, with a twist.  Here, we're going to ask the robots to put everything they find into arbitrary categories based on their characteristics.  Then, they can take us one example from each pile, and we can try looking in the most likely pile(s) for our needles.

* Make us an ultra-precise map.

<img src="/assets/img/unsplash_precise_map.jpg" style="width:500px">

This is a really weird, unintuitive idea.  Instead of have the robots find anything, we're going to have them go out and create a really, *really* nice map.  Then, we can scan the map visually to try and find a needle.  We may even be able to ask the robots to make anything which stands out look really big on the map, like a cartoon treasure map.

# Finding the Needle

At this point, we have three potential strategies for finding our needles.  These ideas align very closely with some ideas in Data Science.

**Anomaly Detection** is a Data Science technique used to find anything that looks out of place in our giant pile of hay - err, data.

**Clustering** is a technique for automatically creating categories of objects, so we can look at one copy of every item.

**Dimensionality-Reduction** (Compression) techniques are used to take giant piles of data and create smaller, easier-to-explore versions of those datasets.

In the next three posts, we're going to review each of these techniques.

In *Honey, I Shrunk the Data*, we're going to review the third of these techniques: map-making.  This technique is actually going to make it easier to perform our other two tasks, since we'll be able to work with much smaller datasets.  In other words, we're going to start by having our robots map out our warehouse full of hay, then have them search through the maps they made.

After that, we'll explore the creation of clusters in *The Brainy Bunch.*

Finally, we'll look at how we can use unsupervised learning to detect anomalies in *Angels in the Outliers*.