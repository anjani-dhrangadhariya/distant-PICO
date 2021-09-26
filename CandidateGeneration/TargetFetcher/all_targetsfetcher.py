#!/usr/bin/env python

def TargetsFetcherDoc(a):
    '''The module retrieves the targets for P.I.C.O.S. sources'''
    return a**a

print( TargetsFetcherDoc.__doc__ )

def fetchIdentificationsTarget(json_document):

    identificationTargets = dict()

    if 'IdentificationModule' in json_document:
        
        if 'BriefTitle' in json_document['IdentificationModule']:
            briefTitle = json_document['IdentificationModule']['BriefTitle']
            identificationTargets['BriefTitle'] = briefTitle

        if 'OfficialTitle' in json_document['IdentificationModule']:
            officialTitle = json_document['IdentificationModule']['OfficialTitle']
            identificationTargets['OfficialTitle'] = officialTitle

    return identificationTargets

def fetchDescriptionsTarget(json_document):

    descriptionTargets = dict()

    if 'DescriptionModule' in json_document:
        
        if 'BriefSummary' in json_document['DescriptionModule']:
            briefSummary = json_document['DescriptionModule']['BriefSummary']
            descriptionTargets['BriefSummary'] = briefSummary

        if 'DetailedDescription' in json_document['DescriptionModule']:
            detailedDescription = json_document['DescriptionModule']['DetailedDescription']
            descriptionTargets['DetailedDescription'] = detailedDescription

    return descriptionTargets


def fetchDesignTarget(json_document):

    designTargets = dict()

    if 'DescriptionModule' in json_document:
        
        if 'DesignInfo' in json_document['DesignModule']:
            designInfo = json_document['DesignModule']['DesignInfo']
            if 'DesignInterventionModelDescription' in designInfo:
                designDescription = designInfo['DesignInterventionModelDescription']
                designTargets['DesignInterventionModelDescription'] = designDescription

    return designTargets


def fetchOutcomesTarget(json_document):

    outcomeTargets = dict()
    outcomeSecondTargets = dict()

    if 'OutcomesModule' in json_document:
            if 'PrimaryOutcomeList' in json_document['OutcomesModule']:
                primOutcome = json_document['OutcomesModule']['PrimaryOutcomeList']['PrimaryOutcome']
                for i, eachOutcome in enumerate(primOutcome):
                    if 'PrimaryOutcomeDescription' in eachOutcome:
                        desc = eachOutcome['PrimaryOutcomeDescription']
                        outcomeTargets['PriOutcomeDesc_' + str(i)] = desc

    if 'OutcomesModule' in json_document:
            if 'SecondaryOutcomeList' in json_document['OutcomesModule']:
                secondOutcome = json_document['OutcomesModule']['SecondaryOutcomeList']['SecondaryOutcome']
                for i, eachOutcome in enumerate(secondOutcome):
                    if 'SecondaryOutcomeDescription' in eachOutcome:
                        desc = eachOutcome['SecondaryOutcomeDescription']
                        outcomeSecondTargets['SecOutcomeDesc_' + str(i)] = desc

    return outcomeTargets, outcomeSecondTargets


def fetchInterventionTargets(json_document):

    interventionTargets = dict()
    armgroupTargets = dict()

    if 'ArmsInterventionsModule' in json_document:
        if 'InterventionList' in json_document['ArmsInterventionsModule']:
            if 'Intervention' in json_document['ArmsInterventionsModule']['InterventionList']:
                intervention = json_document['ArmsInterventionsModule']['InterventionList']['Intervention']

                for i, eachIntervention in enumerate(intervention):
                    if 'InterventionDescription' in eachIntervention:
                        interventionTargets['InterventionDescription_' + str(i)] = eachIntervention['InterventionDescription']

    if 'ArmsInterventionsModule' in json_document:
        if 'ArmGroupList' in json_document['ArmsInterventionsModule']:
            if 'ArmGroup' in json_document['ArmsInterventionsModule']['ArmGroupList']:
                armGroup = json_document['ArmsInterventionsModule']['ArmGroupList']['ArmGroup']

                for i, eachArmGroup in enumerate(armGroup):
                    if 'ArmGroupDescription' in eachArmGroup:
                        armgroupTargets['ArmGroupDescription_' + str(i)] = eachArmGroup['ArmGroupDescription']

    return interventionTargets, armgroupTargets

def fetchEligibilityTarget(json_document):

    eligibilityTargets = dict()

    if 'EligibilityModule' in json_document:
        
        if 'EligibilityCriteria' in json_document['EligibilityModule']:
            eligibility = json_document['EligibilityModule']['EligibilityCriteria']
            eligibilityTargets['EligibilityCriteria'] = eligibility

    return eligibilityTargets

'''
Description:
    Fetches all the targets from the NCT record

Args:
    json_document (json): JSON string containing the protocol_section of a NCT record
        (default is False)

Returns:
    dictionary: a dictionary containing all the targets
'''
def fetchTargets(json_document):

    combined_sources = dict()

    identification_target = fetchIdentificationsTarget(json_document)
    description_target = fetchDescriptionsTarget(json_document)
    design_target = fetchDesignTarget(json_document)
    outcome_target, outcome2_target = fetchOutcomesTarget(json_document)
    intervention_target, armgroup_target = fetchInterventionTargets(json_document)
    eligibility_target = fetchEligibilityTarget(json_document)

    combined_sources = {**identification_target, **description_target, **design_target, **outcome_target, **outcome2_target, **intervention_target, **armgroup_target, **eligibility_target}
    
    return combined_sources