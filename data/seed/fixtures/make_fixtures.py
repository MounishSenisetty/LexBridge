"""Regenerates judgments.jsonl. ALL JUDGMENTS HERE ARE SYNTHETIC TEST FIXTURES, not real cases."""
import json
from pathlib import Path

N = "[SYNTHETIC TEST FIXTURE — not a real judgment]"
J = [
 ("FX-001", "State v. Arjun (synthetic)", "Sessions Court (synthetic)", "2012-03-14", [
  f"{N} The accused is charged under Section 302 of the Indian Penal Code for the murder of the deceased in a land dispute.",
  "Facts: On the night of the incident the accused attacked the deceased with a knife, inflicting injuries on the chest. The deceased died on the way to hospital. Two eye-witnesses deposed to the assault.",
  "The learned counsel for the accused argued that there was no intention to cause death and that the case falls under culpable homicide not amounting to murder.",
  "Analysis: The nature of the injuries, the weapon used and the vital part targeted show the intention to cause death. The ingredients of murder are established beyond reasonable doubt.",
  "Held: The accused is convicted under Section 302 IPC and sentenced to imprisonment for life and fine. The punishment for murder under Section 302 IPC is death or imprisonment for life."]),
 ("FX-002", "Kishan v. State (synthetic)", "High Court (synthetic)", "2015-08-21", [
  f"{N} The appellants challenge their conviction u/s 302/34 for the death of the victim.",
  "Facts: The appellants, acting together, assaulted the victim with sticks after a quarrel. The victim succumbed to head injuries.",
  "Counsel for the appellants submitted that common intention was not proved and the medical evidence was inconsistent.",
  "Analysis: Common intention can be inferred from the conduct of the accused acting in concert. Murder is proved; each appellant shares liability for the murder.",
  "Held: The conviction under Sections 302 and 34 is upheld and the appeal is dismissed. Sentence of life imprisonment for murder is confirmed."]),
 ("FX-003", "Rahim v. State (synthetic)", "Magistrate Court (synthetic)", "2014-01-10", [
  f"{N} The accused is charged under Section 298 IPC for uttering words with deliberate intent to wound the religious feelings of the complainant.",
  "Facts: The accused allegedly made remarks about the complainant's faith in a public gathering.",
  "Counsel for the accused argued that the words were not uttered with a deliberate intention to wound religious feelings.",
  "Analysis: Deliberate intention to wound religious feelings must be proved; mere offence taken by the hearer is not sufficient.",
  "Held: The accused is acquitted of the offence under Section 298 IPC, punishable with imprisonment up to one year or fine."]),
 ("FX-004", "Sunita v. State (synthetic)", "High Court (synthetic)", "2019-06-03", [
  f"{N} The petitioner seeks quashing of the complaint under Section 298 of the Indian Penal Code concerning religious feelings.",
  "Facts: A social media post was alleged to wound the religious feelings of a community.",
  "Analysis: The ingredient of deliberate intention is absent on the face of the complaint.",
  "Held: The complaint under Section 298 IPC is quashed."]),
 ("FX-005", "State v. Manoj (synthetic)", "Sessions Court (synthetic)", "2025-02-11", [
  f"{N} The accused is charged under Section 103(1) of the Bharatiya Nyaya Sanhita, 2023 for the murder of the deceased on 15 August 2024.",
  "Facts: The accused shot the deceased following a business dispute. Ballistic evidence links the weapon to the accused.",
  "The defence argued that the accused was not present at the scene.",
  "Analysis: Since the offence was committed after 1 July 2024, the Bharatiya Nyaya Sanhita governs. Precedents decided under Section 302 IPC on the ingredients of murder remain relevant as Section 103 BNS retains the same punishment for murder.",
  "Held: The accused is convicted under Section 103 BNS and sentenced to imprisonment for life for murder."]),
 ("FX-006", "Imran v. State (synthetic)", "Magistrate Court (synthetic)", "2025-03-20", [
  f"{N} The accused is charged under Section 302 of the Bharatiya Nyaya Sanhita, 2023 for uttering words with deliberate intent to wound the religious feelings of the complainant.",
  "Facts: On 10 October 2024 the accused allegedly made a gesture in front of a place of worship to wound religious feelings.",
  "Analysis: Section 302 BNS punishes deliberate intent to wound religious feelings with simple imprisonment up to one year, or fine, or both.",
  "Held: Charge under Section 302 BNS is framed for wounding religious feelings."]),
 ("FX-007", "Meera v. State (synthetic)", "High Court (synthetic)", "2025-07-01", [
  f"{N} The petitioner seeks quashing of an FIR registered under Section 302 for wounding religious feelings by a video.",
  "Facts: The FIR alleges a video posted in December 2024 wounded the religious feelings of a group.",
  "Analysis: For an offence of wounding religious feelings, deliberate and malicious intention is the core ingredient.",
  "Held: The FIR under Section 302 for wounding religious feelings is quashed for want of deliberate intention."]),
 ("FX-008", "State v. Prakash (synthetic)", "Sessions Court (synthetic)", "2016-11-05", [
  f"{N} The accused is charged under Section 377 IPC for unnatural offences.",
  "Facts: The prosecution alleges non-consensual acts against an adult victim.",
  "Analysis: The ingredients of Section 377 of the Indian Penal Code are discussed with reference to consent.",
  "Held: The accused is convicted under Section 377 IPC."]),
 ("FX-009", "Gupta v. State (synthetic)", "High Court (synthetic)", "2013-04-17", [
  f"{N} The appellant was convicted under Section 420 IPC for cheating investors.",
  "Facts: The appellant induced the complainants to deliver money by promising high returns from a fictitious scheme.",
  "Counsel argued that the dispute was civil in nature and there was no dishonest intention at the inception.",
  "Analysis: Dishonest intention at the time of making the promise is the essence of cheating under Section 420 IPC.",
  "Held: Conviction under Section 420 IPC upheld; sentence of three years imprisonment and fine."]),
 ("FX-010", "Verma v. State (synthetic)", "High Court (synthetic)", "2025-05-09", [
  f"{N} The applicant seeks bail in a case under Section 318(4) of the Bharatiya Nyaya Sanhita for cheating.",
  "Facts: The applicant allegedly induced the complainant in November 2024 to transfer money for a non-existent flat.",
  "Analysis: Cheating under Section 318 BNS requires dishonest inducement to deliver property; earlier precedents under Section 420 IPC on dishonest intention at inception are relevant.",
  "Held: Bail is granted subject to conditions."]),
 ("FX-011", "Lata v. State (synthetic)", "High Court (synthetic)", "2011-09-29", [
  f"{N} The husband and in-laws were convicted under Sections 498A and 304B IPC for cruelty and dowry death.",
  "Facts: The deceased died of burn injuries within four years of marriage; witnesses spoke of demands for dowry soon before her death.",
  "Analysis: The presumption of dowry death arises once cruelty in connection with dowry demand soon before death is shown. Cruelty under Section 498A IPC is independently established.",
  "Held: Convictions under Sections 498-A and 304-B IPC are upheld."]),
 ("FX-012", "Raju v. State (synthetic)", "Sessions Court (synthetic)", "2018-02-12", [
  f"{N} The accused is charged u/s 307 IPC for attempt to murder.",
  "Facts: The accused fired at the complainant, causing injury to the arm.",
  "Analysis: Intention to cause death can be gathered from the use of a firearm aimed at the body; hurt was caused.",
  "Held: The accused is convicted under Section 307 of the Indian Penal Code."]),
 ("FX-013", "Sharma v. State (synthetic)", "Magistrate Court (synthetic)", "2020-10-08", [
  f"{N} The accused is charged under Section 379 IPC for theft of a motorcycle.",
  "Facts: The motorcycle was recovered from the possession of the accused.",
  "Held: The accused is convicted under Section 379 IPC and sentenced to one year imprisonment."]),
 ("FX-014", "Devi v. State (synthetic)", "High Court (synthetic)", "2017-12-01", [
  f"{N} The petitioners seek quashing of proceedings under Section 306 IPC for abetment of suicide.",
  "Analysis: Abetment requires instigation or intentional aiding; a mere quarrel is not instigation.",
  "Held: Proceedings under Section 306 IPC are quashed."]),
]

GOLD = {
 "FX-001": ["IPC_302"], "FX-002": ["IPC_302", "IPC_34"], "FX-003": ["IPC_298"], "FX-004": ["IPC_298"],
 "FX-005": ["BNS_103", "IPC_302"], "FX-006": ["BNS_302"], "FX-007": ["BNS_302"], "FX-008": ["IPC_377"],
 "FX-009": ["IPC_420"], "FX-010": ["BNS_318", "IPC_420"], "FX-011": ["IPC_498A", "IPC_304B"],
 "FX-012": ["IPC_307"], "FX-013": ["IPC_379"], "FX-014": ["IPC_306"],
}

out = Path(__file__).parent / "judgments.jsonl"
with open(out, "w") as f:
    for doc_id, title, court, date, paras in J:
        f.write(json.dumps({"doc_id": doc_id, "title": title, "court": court, "date": date,
                            "text": "\n\n".join(paras), "source": "fixture", "synthetic": True,
                            "gold_sections": GOLD[doc_id]}) + "\n")
print(f"wrote {len(J)} fixtures to {out}")
